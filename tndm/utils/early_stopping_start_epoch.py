import tensorflow as tf

class EarlyStoppingStartEpoch(tf.keras.callbacks.Callback):
    def __init__(self, 
                 monitor='val_loss/reconstruction', 
                 min_delta=0, 
                 patience=10, 
                 verbose=1, 
                 mode='min', 
                 baseline=None, 
                 restore_best_weights=False, 
                 start_from_epoch=0):
        super(EarlyStoppingStartEpoch, self).__init__()

        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

        if mode == 'min':
            self.monitor_op = lambda current, best: current < best - min_delta
            self.best = float('inf')
        elif mode == 'max':
            self.monitor_op = lambda current, best: current > best + min_delta
            self.best = float('-inf')
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'min' or 'max'.")

    def on_train_begin(self, logs=None):
        if self.baseline is not None:
            self.best = self.baseline

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        
        if current is None:
            print(f"Early stopping requires {self.monitor} available!")
            return
        
        # If we are before the start_from_epoch, do nothing
        if epoch < self.start_from_epoch:
            return

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose > 0:
                        print(f"Restoring model weights from the end of the best epoch: {self.best}")
                    self.model.set_weights(self.best_weights)
        
        if self.verbose > 0 and self.model.stop_training:
            print(f"Epoch {epoch + 1}: early stopping")

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping triggered")
