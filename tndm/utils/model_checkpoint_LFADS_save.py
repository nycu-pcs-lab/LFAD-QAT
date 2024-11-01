import tensorflow as tf

class ModelCheckpointLFADSSave(tf.keras.callbacks.Callback):
    def __init__(self, filepath, save_best_only=True, monitor='val_loss/reconstruction', mode='auto'):
        super(ModelCheckpointLFADSSave, self).__init__()
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.best = None
        self.monitor_op = None

        if mode == 'min':
            self.monitor_op = tf.math.less
            self.best = float('inf')
        elif mode == 'max':
            self.monitor_op = tf.math.greater
            self.best = float('-inf')
        else:
            if 'acc' in self.monitor or 'accuracy' in self.monitor:
                self.monitor_op = tf.math.greater
                self.best = float('-inf')
            else:
                self.monitor_op = tf.math.less
                self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        filepath = self.filepath
        current = logs.get(self.monitor)

        if current is None:
            return

        if self.save_best_only:
            if self.monitor_op(current, self.best):
                previous_best = self.best
                self.best = current
                self.model.save(filepath)
                print(f"Epoch {epoch + 1}: {self.monitor} improved from {previous_best:.5f} to {current:.5f}, saving model to {filepath}")
            else:
                print(f"Epoch {epoch + 1}: {self.monitor} did not improve from {self.best:.5f}")
        else:
            self.model.save(filepath)
            print(f"Epoch {epoch + 1}: {self.monitor} was {current:.5f}, saving model to {filepath}")
