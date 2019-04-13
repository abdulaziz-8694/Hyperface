from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.losses import categorical_crossentropy, mean_squared_error
import tensorflow as tf

from config import Config
from data.loader import DataLoader
from model.architecure import HyperFaceNetworkAlexNet, HyperFaceNetworkResNet


class Trainer:
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.checkpoint_path = config.checkpoint_path
        self.model_path = config.model_save_path
        self.epochs = config.epochs
        self.arch_type = config.arch_type
        self.sample_data_file = config.sample_data_file
        self.input_shape = config.input_shape
        self.learning_rate = config.learning_rate
        self.data_loader = DataLoader(self.sample_data_file)

    def _get_model(self):
        if self.arch_type == "alexNet":
            return HyperFaceNetworkAlexNet(self.input_shape).model
        elif self.arch_type == "resNet":
            return HyperFaceNetworkResNet(self.input_shape).model

    def get_losses(self):
        def custom_gender_loss():
            def custom_loss(y_true, y_pred):
                # Ignoring outputs where the face ground thruth us negativce.
                # Since only two classes are valid, the loss is calculated for those logits
                boolean_mask = tf.not_equal(y_pred, tf.Variable([0, 0, 0], dtype=tf.float32))
                y_true_filtered = tf.boolean_mask(y_true, boolean_mask)
                y_pred_filtered = tf.boolean_mask(y_pred, boolean_mask)
                return categorical_crossentropy(y_true_filtered[-2:], y_pred_filtered[-2:])

            return custom_loss

        def custom_mse_error(n):
            def custom_loss(y_true, y_pred):
                # Ignoring outputs where the face ground thruth us negativce.
                boolean_mask = tf.not_equal(y_pred, tf.Variable([0 for _ in range(n)], dtype=tf.float32))
                y_true_filtered = tf.boolean_mask(y_true, boolean_mask)
                y_pred_filtered = tf.boolean_mask(y_pred, boolean_mask)
                return mean_squared_error(y_true_filtered, y_pred_filtered)

            return custom_loss

        loss = {
            "face_logits": "sparse_categorical_crossentropy",
            "landmark_logits": custom_mse_error(42),
            "visibility_logits": custom_mse_error(21),
            "pose_logits": custom_mse_error(3),
            "gender_logits": custom_gender_loss(),
        }
        # Assigning weights for each task loss according to paper.
        loss_weights = {
            "face_logits": 1,
            "landmark_logits": 5,
            "visibility_logits": 0.5,
            "pose_logits": 5,
            "gender_logits": 2,
        }
        return loss, loss_weights

    def get_callbacks(self):
        checkpoint_cb = ModelCheckpoint(
            self.checkpoint_path, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1
        )
        earlystopping_cb = EarlyStopping(monitor="val_loss", patience=5, verbose=1)
        tensorboard = TensorBoard(log_dir="./events/")
        return [checkpoint_cb, earlystopping_cb, tensorboard]

    def train(self):
        model = self._get_model()
        model.summary()
        loss, loss_weights = self.get_losses()
        callbacks = self.get_callbacks()
        model.compile(optimizer=Adam(lr=self.learning_rate), loss=loss, loss_weights=loss_weights, metrics=["accuracy"])
        # visibility and face det is added as input as to mask the 4 tasks logits
        model.fit(
            x=[
                self.data_loader.X,
                self.data_loader.expand_visibility(self.data_loader.labels[2], 2),
                self.data_loader.labels[0],
            ],
            y=self.data_loader.labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            callbacks=callbacks,
            validation_split=0.2,
            shuffle=True,
        )
        # saving whole model
        model.save(self.model_path)


if __name__ == "__main__":
    config = Config()
    trainer = Trainer(config)
    trainer.train()
