from tensorflow.keras.callbacks import Callback
from sklearn.metrics import matthews_corrcoef


class MCCMetricCallback(Callback):

    def __init__(self, val_data, beta=1.0, threshold=0.5):
        super().__init__()
        self.validation_data = val_data
        self.beta = beta
        self.threshold = threshold
        # Will be initialized when the training starts
        self.val_mcc = None

    def on_train_begin(self, logs=None):
        """ This is where the validation Fbeta
        validation scores will be saved during training: one value per
        epoch.
        """
        self.val_mcc = []

    def on_epoch_end(self, epoch, logs={}):
        val_predictions = self.model.predict(self.validation_data[0])
        val_targets = self.validation_data[1]

        thresholded_predictions = (val_predictions > self.threshold).astype(int)

        val_mcc = matthews_corrcoef(val_targets, thresholded_predictions)
        self.val_mcc.append(val_mcc)

        logs["val_mcc"] = val_mcc

        # print("- val_f1_score: {:.4f}".format(val_mcc))
        return

    def on_train_end(self, logs=None):
        """ Assign the validation Fbeta computed metric to the History object.
        """
        self.model.history.history["val_mcc"] = self.val_mcc

