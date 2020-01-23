from tensorflow.keras.callbacks import Callback
from sklearn.metrics import fbeta_score


class FBetaMetricCallback(Callback):

    def __init__(self, val_data, beta=1.0, threshold=0.5):
        super().__init__()
        self.validation_data = val_data
        self.beta = beta
        self.threshold = threshold
        # Will be initialized when the training starts
        self.val_fbeta = None

    def on_train_begin(self, logs=None):
        """ This is where the validation Fbeta
        validation scores will be saved during training: one value per
        epoch.
        """
        self.val_fbeta = []

    def _score_per_threshold(self, predictions, targets, threshold):
        """ Compute the Fbeta score per threshold.
        """
        # Notice that here I am using the sklearn fbeta_score function.
        # You can read more about it here:
        # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html
        thresholded_predictions = (predictions > threshold).astype(int)
        return fbeta_score(targets, thresholded_predictions, beta=self.beta, average='weighted')

    def on_epoch_end(self, epoch, logs={}):
        val_predictions = self.model.predict(self.validation_data[0])
        val_targets = self.validation_data[1]
        _val_fbeta = self._score_per_threshold(val_predictions, val_targets, self.threshold)
        self.val_fbeta.append(_val_fbeta)

        logs["val_f1_score"] = _val_fbeta

        print("- val_f1_score: {:.4f}".format(_val_fbeta))
        return

    def on_train_end(self, logs=None):
        """ Assign the validation Fbeta computed metric to the History object.
        """
        self.model.history.history["val_fbeta"] = self.val_fbeta

