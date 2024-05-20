from __future__ import annotations
from copy import deepcopy
import tensorflow as tf
from typing import Dict, Any
from collections import defaultdict

from tndm.utils import ArgsParser, clean_layer_name, logger
from tndm.layers import GaussianSampling, GeneratorGRU
from tndm.losses import gaussian_kldiv_loss, poisson_loglike_loss, regularization_loss
from .model_loader import ModelLoader


tf.config.run_functions_eagerly(True)

# quantize
from qkeras import QGRU, QDense, QBidirectional, QActivation

class LFADS(ModelLoader, tf.keras.Model):

    def __init__(self, **kwargs: Dict[str, Any]):
        tf.keras.Model.__init__(self)

        self.full_logs: bool = bool(ArgsParser.get_or_default(
            kwargs, 'full_logs', False))
        self.encoder_dim: int = int(ArgsParser.get_or_default(
            kwargs, 'encoder_dim', 64))
        self.initial_condition_dim: int = int(ArgsParser.get_or_default(
            kwargs, 'initial_condition_dim', 64))
        self.decoder_dim: int = int(ArgsParser.get_or_default(
            kwargs, 'decoder_dim', 64))
        self.factors: int = int(ArgsParser.get_or_default(kwargs, 'factors', 3))
        self.neural_dim: int = int(ArgsParser.get_or_error(
            kwargs, 'neural_dim'))
        self.max_grad_norm: float = float(ArgsParser.get_or_default(
            kwargs, 'max_grad_norm', 200))
        self.timestep: float = float(ArgsParser.get_or_default(
            kwargs, 'timestep', 0.01))
        self.prior_variance: float = float(ArgsParser.get_or_default(
            kwargs, 'prior_variance', 0.1))
        self.dropout: float = float(ArgsParser.get_or_default(
            kwargs, 'dropout', 0.05))
        self.with_behaviour = False
        self.neural_lik_type: str = str(ArgsParser.get_or_default(
            kwargs, 'neural_lik_type','poisson'))
        self.threshold_poisson_log_firing_rate: float = float(ArgsParser.get_or_default(
            kwargs, 'threshold_poisson_log_firing_rate', 100.0))
        self.GRU_pre_activation: bool = bool(ArgsParser.get_or_default(
            kwargs, 'GRU_pre_activation', False))

        self.neural_loglike_loss = poisson_loglike_loss(self.timestep)
        
        self.quantized: bool = bool(ArgsParser.get_or_default(
            kwargs, 'quantized', False))
        self.encoder_quantized: bool = bool(ArgsParser.get_or_default(
            kwargs, 'encoder_quantized', False))
        self.total_bit: int = int(ArgsParser.get_or_default(
            kwargs, 'total_bit', 16))
        self.int_bit_weight: int = int(ArgsParser.get_or_default(
            kwargs, 'int_bit_weight', 1))
            
        # for 16, 24, 12, 10
        self.int_bit: int = int(ArgsParser.get_or_default(
            kwargs, 'int_bit', 6))
        # for 8, 6, 4
        self.int_bit_big: int = int(ArgsParser.get_or_default(
            kwargs, 'int_bit_big', 6))
        self.int_bit_small: int = int(ArgsParser.get_or_default(
            kwargs, 'int_bit_small', 2))
            
        self.high_bit = bool(self.total_bit > 8)
            
        # Quantize
        # For higher bit, there is no difference between act_quan_big and act_quan_small
        if self.high_bit:
            self.act_quan_big = "quantized_bits({},{},alpha=1)".format(self.total_bit, self.int_bit)
            self.act_quan_small = "quantized_bits({},{},alpha=1)".format(self.total_bit, self.int_bit)
            self.gru_quan = "quantized_bits({},{},alpha=1)".format(self.total_bit, self.int_bit_weight)
            self.act_quan_sigmoid = "quantized_sigmoid({})".format(self.total_bit)
            self.act_quan_tanh = "quantized_tanh({})".format(self.total_bit)
        else:
            self.act_quan_big = "quantized_bits({},{},alpha=1)".format(16, self.int_bit_big)
            self.act_quan_small = "quantized_bits({},{},alpha=1)".format(self.total_bit, self.int_bit_small)
            self.gru_quan = "quantized_bits({},{},alpha=1)".format(self.total_bit, self.int_bit_weight)
            self.act_quan_sigmoid = "quantized_sigmoid({})".format(self.total_bit)
            self.act_quan_tanh = "quantized_tanh({})".format(self.total_bit)
            
        layers = ArgsParser.get_or_default(kwargs, 'layers', {})
        if not isinstance(layers, defaultdict):
            layers: Dict[str, Any] = defaultdict(
                lambda: dict(
                    kernel_regularizer=tf.keras.regularizers.L2(l=1),
                    kernel_initializer=tf.keras.initializers.VarianceScaling(distribution='normal')),
                layers
            )
        self.layers_settings = deepcopy(layers)

        # METRICS
        self.tracker_loss = tf.keras.metrics.Sum(name="loss")
        self.tracker_loss_loglike = tf.keras.metrics.Sum(name="loss_loglike")
        self.tracker_loss_kldiv = tf.keras.metrics.Sum(name="loss_kldiv")
        self.tracker_loss_reg = tf.keras.metrics.Sum(name="loss_reg")
        self.tracker_loss_count = tf.keras.metrics.Sum(name="loss_count")
        self.tracker_loss_w_loglike = tf.keras.metrics.Mean(
            name="loss_w_loglike")
        self.tracker_loss_w_kldiv = tf.keras.metrics.Mean(name="loss_w_kldiv")
        self.tracker_loss_w_reg = tf.keras.metrics.Mean(name="loss_w_reg")
        self.tracker_lr = tf.keras.metrics.Mean(name="lr")

        # ENCODER
        self.initial_dropout = tf.keras.layers.Dropout(self.dropout)
        encoder_args: Dict[str, Any] = layers['encoder']
        self.encoded_var_min: float = ArgsParser.get_or_default_and_remove(
            encoder_args, 'var_min', .0001)
        self.encoded_var_trainable: bool = ArgsParser.get_or_default_and_remove(
            encoder_args, 'var_trainable', True)
        if self.quantized or self.encoder_quantized:
            forward_layer = QGRU(
                self.encoder_dim, time_major=False, name="EncoderGRUForward", return_state=False, reset_after=True, **encoder_args,
                activation=self.act_quan_tanh,
                recurrent_activation=self.act_quan_sigmoid,
                kernel_quantizer=self.gru_quan,
                recurrent_quantizer=self.gru_quan,
                bias_quantizer=self.gru_quan,
                state_quantizer=self.act_quan_small)
            backward_layer = QGRU(
                self.encoder_dim, time_major=False, name="EncoderGRUBackward", return_state=False, reset_after=True, go_backwards=True, **encoder_args,
                activation=self.act_quan_tanh,
                recurrent_activation=self.act_quan_sigmoid,
                kernel_quantizer=self.gru_quan,
                recurrent_quantizer=self.gru_quan,
                bias_quantizer=self.gru_quan,
                state_quantizer=self.act_quan_small)
            self.encoder = QBidirectional(
                forward_layer, backward_layer=backward_layer, name='EncoderRNN', merge_mode='concat')  
            self.q_act_postencoder = QActivation(self.act_quan_small, name = "q_act_postencoder")
        else:
            forward_layer = tf.keras.layers.GRU(
                self.encoder_dim, time_major=False, name="EncoderGRUForward", return_state=False, **encoder_args)
            backward_layer = tf.keras.layers.GRU(
                self.encoder_dim, time_major=False, name="EncoderGRUBackward", return_state=False, go_backwards=True, **encoder_args)
            self.encoder = tf.keras.layers.Bidirectional(
                forward_layer, backward_layer=backward_layer, name='EncoderRNN', merge_mode='concat')
        self.dropout_post_encoder = tf.keras.layers.Dropout(self.dropout)
        self.dropout_post_decoder = tf.keras.layers.Dropout(self.dropout)
        
        # DISTRIBUTION
        if self.quantized:
            self.dense_mean = QDense(
                self.initial_condition_dim,
                kernel_quantizer=self.gru_quan,
                bias_quantizer=self.gru_quan,
                name="DenseMean", **layers['dense_mean'])
            self.dense_logvar = QDense(
                self.initial_condition_dim,
                kernel_quantizer=self.gru_quan,
                bias_quantizer=self.gru_quan,
                name="DenseLogVar", **layers['dense_logvar'])
            self.q_act_dense_mean = QActivation(self.act_quan_big, name = "q_act_dense_mean")
            self.q_act_dense_logvar = QActivation(self.act_quan_small, name = "q_act_dense_logvar")
        else:
            self.dense_mean = tf.keras.layers.Dense(
                self.initial_condition_dim, name="DenseMean", **layers['dense_mean'])
            self.dense_logvar = tf.keras.layers.Dense(
                self.initial_condition_dim, name="DenseLogVar", **layers['dense_logvar'])

        # SAMPLING
        self.sampling = GaussianSampling(name="GaussianSampling")
        if self.quantized:
            self.q_act_postsampling = QActivation(self.act_quan_big, name = "q_act_postsampling")

        # DECODERS
        if self.decoder_dim != self.initial_condition_dim:
            self.dense_pre_decoder = tf.keras.layers.Dense(
                self.decoder_dim, name="DensePreDecoder", **layers['dense_pre_decoder'])
        self.pre_decoder_activation = tf.keras.layers.Activation('tanh')
        decoder_args: Dict[str, Any] = layers['decoder']
        self.original_generator: float = ArgsParser.get_or_default_and_remove(
            decoder_args, 'original_cell', False)
        if self.original_generator:
            decoder_cell = GeneratorGRU(self.decoder_dim, **decoder_args)
            self.decoder = tf.keras.layers.RNN(
                decoder_cell, return_sequences=True, time_major=False, name='DecoderGRU')
        else:
            if self.quantized:
                self.decoder = QGRU(
                    self.decoder_dim, return_sequences=True, time_major=False, reset_after=True, name='DecoderGRU', **decoder_args,
                    activation=self.act_quan_tanh,
                    recurrent_activation=self.act_quan_sigmoid,
                    kernel_quantizer=self.gru_quan,
                    recurrent_quantizer=self.gru_quan,
                    bias_quantizer=self.gru_quan,
                    state_quantizer=self.act_quan_small)
                self.q_act_predecoder = QActivation(self.act_quan_big, name = "q_act_predecoder")
                self.q_act_postdecoder = QActivation(self.act_quan_big, name = "q_act_postdecoder")
            else:
                self.decoder = tf.keras.layers.GRU(
                    self.decoder_dim, return_sequences=True, time_major=False, name='DecoderGRU', **decoder_args)

        # DIMENSIONALITY REDUCTION
        if self.quantized:
            self.dense = QDense(
                self.factors, use_bias=False,
                kernel_quantizer=self.gru_quan,
                bias_quantizer=self.gru_quan,
                name="Dense", **layers['dense'])
            self.q_act_postdense = QActivation(self.act_quan_small, name = "q_act_postdense")
        else:
            self.dense = tf.keras.layers.Dense(
                self.factors, use_bias=False, name="Dense", **layers['dense'])

        # NEURAL
        if self.quantized:
            self.neural_dense = QDense(
                self.neural_dim,
                kernel_quantizer=self.gru_quan,
                bias_quantizer=self.gru_quan,
                name="NeuralDense", **layers['neural_dense'])
        else:
            self.neural_dense = tf.keras.layers.Dense(
                self.neural_dim, name="NeuralDense", **layers['neural_dense'])

    @staticmethod
    def load(filename) -> LFADS:
        return ModelLoader.load(filename, LFADS)

    def get_settings(self):
        return dict(        
            neural_lik_type=self.neural_lik_type,
            encoder_dim=self.encoder_dim,
            decoder_dim=self.decoder_dim,
            initial_condition_dim=self.initial_condition_dim,
            factors=self.factors,
            neural_dim=self.neural_dim,
            max_grad_norm=self.max_grad_norm,
            timestep=self.timestep,
            prior_variance=self.prior_variance,
            layers=self.layers_settings,
            default_layer_settings=self.layers_settings.default_factory(),
            full_logs=self.full_logs,
            quantized=self.quantized,
            encoder_quantized=self.encoder_quantized,
            total_bit=self.total_bit,
            int_bit_weight=self.int_bit_weight,
            int_bit=self.int_bit,
            int_bit_big=self.int_bit_big,
            int_bit_small=self.int_bit_small
        )
    
    def load_model_weight(self, source_LFAD_model):
        self.encoder.set_weights(source_LFAD_model.get_layer("EncoderRNN").get_weights())
        self.dense_mean.set_weights(source_LFAD_model.get_layer("DenseMean").get_weights())
        self.dense_logvar.set_weights(source_LFAD_model.get_layer("DenseLogVar").get_weights())
        self.decoder.set_weights(source_LFAD_model.get_layer("DecoderGRU").get_weights())
        self.dense.set_weights(source_LFAD_model.get_layer("Dense").get_weights())
        self.neural_dense.set_weights(source_LFAD_model.get_layer("NeuralDense").get_weights())

    @tf.function
    def call(self, inputs, training: bool = True):
        g0, mean, logvar = self.encode(inputs, training=training)
        log_f, z = self.decode(g0, inputs, training=training)
        return log_f, (g0, mean, logvar), z
    
    @tf.function
    def encode(self, inputs, training: bool = True):
        dropped_neural = self.initial_dropout(inputs, training=training)
        encoded = self.encoder(dropped_neural, training=training)
        if self.quantized or self.encoder_quantized:
            encoded = self.q_act_postencoder(encoded)
        dropped_encoded = self.dropout_post_encoder(encoded, training=training)

        mean = self.dense_mean(dropped_encoded, training=training)
        if self.quantized:
            qmean = self.q_act_dense_mean(mean)

        if self.encoded_var_trainable:
            logvar = tf.math.log(tf.exp(self.dense_logvar(
                dropped_encoded, training=training)) + self.encoded_var_min)
        else:
            logvar = tf.zeros_like(mean) + tf.math.log(self.encoded_var_min)
        
        if self.quantized:
            qlogvar = self.q_act_dense_logvar(logvar)
        if self.quantized:
            g0 = self.sampling(
                tf.stack([qmean, qlogvar], axis=-1), training=training)
        else:
            g0 = self.sampling(
                tf.stack([mean, logvar], axis=-1), training=training)
        if self.quantized:
            g0 = self.q_act_postsampling(g0)
        return g0, mean, logvar

    @tf.function
    def decode(self, g0, inputs, training: bool = True):
        # Assuming inputs are zero and everything comes from the GRU
        u = tf.stack([tf.zeros_like(inputs)[:, :, -1]
                     for i in range(self.decoder.cell.units)], axis=-1)

        if self.decoder_dim != self.initial_condition_dim:
            g0 = self.dense_pre_decoder(g0, training=training)
        if self.GRU_pre_activation:
            g0_pre_decoder = self.pre_decoder_activation(g0) # Not in the original
        else:
            g0_pre_decoder = g0
        
        if self.quantized:
            g0_pre_decoder = self.q_act_predecoder(g0_pre_decoder)
        
        g = self.decoder(u, initial_state=g0_pre_decoder, training=training)
        if self.quantized:
            g = self.q_act_postdecoder(g)
        
        dropped_g = self.dropout_post_decoder(g, training=training) #dropout after GRU
        z = self.dense(dropped_g, training=training)
        
        if self.quantized:
            z = self.q_act_postdense(z)

        # clipping the log-firingrate log(self.timestep) so that the
        # log-likelihood does not return NaN
        # (https://github.com/tensorflow/tensorflow/issues/47019)
        if self.neural_lik_type == 'poisson':
            log_f = tf.clip_by_value(self.neural_dense(z, training=training), 
                                     clip_value_min=-self.threshold_poisson_log_firing_rate,
                                     clip_value_max=self.threshold_poisson_log_firing_rate)
        else:
            log_f = self.neural_dense(z, training=training)

        # In order to be able to auto-encode, the dimensions should be the same
        if not self.built:
            assert all([f_i == i_i for f_i, i_i in zip(
                list(log_f.shape), list(inputs.shape))])

        return log_f, z

    def compile(self, optimizer, loss_weights, *args, **kwargs):
        super(LFADS, self).compile(
            loss=[
                poisson_loglike_loss(self.timestep),
                gaussian_kldiv_loss(self.prior_variance),
                regularization_loss()],
            optimizer=optimizer,
        )
        self.loss_weights = loss_weights

        if self.full_logs:
            self.tracker_gradient_dict = {'grads/' + clean_layer_name(x.name):
                                        tf.keras.metrics.Sum(name=clean_layer_name(x.name)) for x in
                                        self.trainable_variables if 'bias' not in x.name.lower()}
            self.tracker_norms_dict = {'norms/' + clean_layer_name(x.name):
                                    tf.keras.metrics.Sum(name=clean_layer_name(x.name)) for x in
                                    self.trainable_variables if 'bias' not in x.name.lower()}
            self.tracker_batch_count = tf.keras.metrics.Sum(name="batch_count")

    @tf.function
    def train_step(self, data):
        """The logic for one training step.
        This method can be overridden to support custom training logic.
        For concrete examples of how to override this method see
        [Customizing what happends in fit](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit).
        This method is called by `Model.make_train_function`.
        This method should contain the mathematical logic for one step of training.
        This typically includes the forward pass, loss calculation, backpropagation,
        and metric updates.
        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.
        Args:
          data: A nested structure of `Tensor`s.
        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided.
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            log_f, g, _ = self(x, training=True)

            loss_loglike    = self.compiled_loss._losses[0](log_f,x) 
            loss_kldiv      = self.compiled_loss._losses[1](g)
            loss_reg        = self.compiled_loss._losses[2](self.losses)

            loss = self.loss_weights[0] * loss_loglike + \
                self.loss_weights[1] * loss_kldiv + \
                self.loss_weights[2] * loss_reg
            unclipped_grads = tape.gradient(loss, self.trainable_variables)

        # For numerical stability (clip_by_global_norm returns NaNs for large
        # grads, becaues grad_global_norms goes to Inf)
        value_clipped_grads = [tf.clip_by_value(
            x, -1e16, 1e16) if x is not None else x for x in unclipped_grads]
        grads, grad_global_norm = tf.clip_by_global_norm(
            value_clipped_grads, self.max_grad_norm)
        # Run backwards pass.

        self.optimizer.apply_gradients(
            (grad, var)
            for (grad, var) in zip(grads, self.trainable_variables)
            if grad is not None
        )

        # Compute our own metrics
        self.tracker_loss.update_state(loss)
        self.tracker_loss_loglike.update_state(loss_loglike)
        self.tracker_loss_kldiv.update_state(loss_kldiv)
        self.tracker_loss_reg.update_state(loss_reg)
        self.tracker_loss_w_loglike.update_state(self.loss_weights[0])
        self.tracker_loss_w_kldiv.update_state(self.loss_weights[1])
        self.tracker_loss_w_reg.update_state(self.loss_weights[2])
        self.tracker_lr.update_state(
            self.optimizer._decayed_lr('float32').numpy())
        self.tracker_loss_count.update_state(x.shape[0])

        core_logs = {'loss': self.tracker_loss.result() / self.tracker_loss_count.result(),
                'loss/loglike': self.tracker_loss_loglike.result() / self.tracker_loss_count.result(),
                'loss/kldiv': self.tracker_loss_kldiv.result() / self.tracker_loss_count.result(),
                'loss/reg': self.tracker_loss_reg.result(),
                'loss/reconstruction': self.tracker_loss_loglike.result() / self.tracker_loss_count.result(),
                'weights/loglike': self.tracker_loss_w_loglike.result(),
                'weights/kldiv': self.tracker_loss_w_kldiv.result(),
                'weights/reg': self.tracker_loss_w_reg.result(),
                'learning_rate': self.tracker_lr.result()}

        if self.full_logs:
            self.tracker_batch_count.update_state(1)

            for grad, var in zip(grads, self.trainable_variables):
                if 'bias' not in var.name.lower():
                    cleaned_name = clean_layer_name(var.name)
                    self.tracker_gradient_dict['grads/' +
                                            cleaned_name].update_state(tf.norm(grad, 1))
                    self.tracker_norms_dict['norms/' +
                                            cleaned_name].update_state(tf.norm(var, 1))

            return {
                **core_logs
                **{k: v.result() / self.tracker_batch_count.result() for k, v in self.tracker_gradient_dict.items()},
                **{k: v.result() / self.tracker_batch_count.result() for k, v in self.tracker_norms_dict.items()}
            }
        else:
            return core_logs

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        core_losses = [
            self.tracker_loss,
            self.tracker_loss_loglike,
            self.tracker_loss_kldiv,
            self.tracker_loss_reg,
            self.tracker_loss_w_loglike,
            self.tracker_loss_w_kldiv,
            self.tracker_loss_w_reg,
            self.tracker_lr,
            self.tracker_loss_count,
        ]
        if self.full_logs:
            return core_losses + [self.tracker_batch_count] + list(self.tracker_norms_dict.values()) + list(self.tracker_gradient_dict.values())
        else:
            return core_losses

    @tf.function
    def test_step(self, data):
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided.
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Run forward pass.
        log_f, g, _ = self(x, training=False)

        loss_loglike    = self.compiled_loss._losses[0](log_f,x)  
        loss_kldiv      = self.compiled_loss._losses[1](g)

        loss = self.loss_weights[0] * loss_loglike + \
            self.loss_weights[1] * loss_kldiv

        # Update the metrics.
        self.tracker_loss.update_state(loss)
        self.tracker_loss_loglike.update_state(loss_loglike)
        self.tracker_loss_kldiv.update_state(loss_kldiv)
        self.tracker_loss_count.update_state(x.shape[0])

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {
            'loss': self.tracker_loss.result() / self.tracker_loss_count.result(),
            'loss/loglike': self.tracker_loss_loglike.result() / self.tracker_loss_count.result(),
            'loss/kldiv': self.tracker_loss_kldiv.result() / self.tracker_loss_count.result(),
            'loss/reconstruction': self.tracker_loss_loglike.result() / self.tracker_loss_count.result(),
        }
