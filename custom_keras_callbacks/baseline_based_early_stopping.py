from keras.src.callbacks.early_stopping import EarlyStopping

class BaselineBasedEarlyStopping(EarlyStopping):
    def __init__(
        self,
        baseline,
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        restore_best_weights=False,
        start_from_epoch=0
    ):
        # Call __init__ from the parent class
        # with baseline modified to have no default value
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            baseline=baseline,
            restore_best_weights=restore_best_weights,
            start_from_epoch=start_from_epoch
        )

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.best_weights = None
        self.best_epoch = 0
        self.has_surpassed_baseline = False

    def on_epoch_end(self, epoch, logs=None):
        if self.monitor_op is None:
            # Delay setup until the model's metrics are all built
            self._set_monitor_op()

        current = self.get_monitor_value(logs)
        if current is None or epoch < self.start_from_epoch:
            # If no monitor value exists or still in initial warm-up stage.
            return
        if self.restore_best_weights and self.best_weights is None:
            # If best weights were never set,
            # then the current weights are the best.
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch
        
        is_improvement = self._is_improvement(current, self.best)
        if is_improvement:
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()

        # Start early stopping wait counter if performance has surpassed baseline,
        # and keep increasing wait counter if baseline was surpassed before but not now
        if self.monitor_op(current, self.baseline) or self.has_surpassed_baseline:
            self.has_surpassed_baseline = True
            self.wait += 1
            # Only restart wait counter if we beat our previous best
            if is_improvement:
                self.wait = 0
                return
        else:
            # If the baseline has never been surpassed at all,
            # do nothing and continue training
            return

        if self.wait >= self.patience and epoch > 0:
            # Patience has been exceeded: stop training
            self.stopped_epoch = epoch
            self.model.stop_training = True
