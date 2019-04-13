from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Concatenate, Flatten, Multiply, Lambda


class HyperFaceNetworkAlexNet(object):
    """
    This creates the architecture of the network using keras APIs. For the case of custom loss of landmarks
    The logits are masked with visibility feature since in loss computation those with visibility 0 will not
    have any impact, hence after masking the logit and input for landmark with visibility, loss is equivalent
    to mean squared error. Similarly for the tasks other than face detection, the loss should be calculated
    for outputs where the face detection ground truth is positive. This is encoded by masking face detection
    ground truth for every logits and then during loss calculation these outputs will be filtered out.
    """

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None
        self._create_graph()

    def _create_graph(self):
        # creating alex net architecture for the first half of network
        input_tensor = Input(shape=self.input_shape, name="INPUT")
        # Mask for
        visibility_mask = Input(shape=[42], name="visibility_mask")
        detection_mask = Input(shape=[1], name="detection_mask")
        conv1 = Conv2D(filters=96, kernel_size=[11, 11], strides=[4, 4], activation="relu", name="conv1")(input_tensor)
        max_pool1 = MaxPooling2D(pool_size=[3, 3], strides=[2, 2], name="max_pool1")(conv1)
        conv2 = Conv2D(filters=256, kernel_size=[5, 5], padding="same", activation="relu", name="conv2")(max_pool1)
        max_pool2 = MaxPooling2D(pool_size=[3, 3], strides=[2, 2], name="max_pool2")(conv2)
        conv3 = Conv2D(filters=384, kernel_size=[3, 3], padding="same", activation="relu", name="conv3")(max_pool2)
        conv4 = Conv2D(filters=384, kernel_size=[3, 3], padding="same", activation="relu", name="conv4")(conv3)
        conv5 = Conv2D(filters=256, kernel_size=[3, 3], padding="same", activation="relu", name="conv5")(conv4)
        max_pool5 = MaxPooling2D(pool_size=[3, 3], strides=[2, 2], name="max_pool5")(conv5)

        # Extracting different level features from different layers
        conv1a = Conv2D(filters=256, kernel_size=[4, 4], strides=[4, 4], activation="relu", name="conv1a")(max_pool1)
        conv3a = Conv2D(filters=256, kernel_size=[2, 2], strides=[2, 2], activation="relu", name="conv3a")(conv3)

        # Concatenating the different features extracted
        concat_feat = Concatenate(name="concat_feat")([conv1a, conv3a, max_pool5])

        # Dimensionality reduction and flattening
        conv_final = Conv2D(filters=192, kernel_size=[1, 1], activation="relu", name="conv_final")(concat_feat)
        flatten = Flatten(name="flatten")(conv_final)
        fc_full = Dense(units=3072, activation="relu", name="fc_full")(flatten)

        # Branching into five outputs for five tasks
        face_fc = Dense(units=512, activation="relu", name="face_fc")(fc_full)
        face_logits = Dense(units=2, name="face_logits")(face_fc)

        # Masking logits of every other task based on the ground truth of face detection
        # This is like an encoding which will used to filter outputs where face groundtruth
        # was negative
        landmark_fc = Dense(units=512, activation="relu", name="landmark_fc")(fc_full)
        landmark_logits_unmasked = Dense(units=42, name="landmark_logits_unmasked")(landmark_fc)
        landmark_logits_masked = Multiply(name="landmark_logits_masked")([landmark_logits_unmasked, visibility_mask])
        landmark_logits = Multiply(name="landmark_logits")([landmark_logits_masked, detection_mask])
        visibility_fc = Dense(units=512, activation="relu", name="visibiity_fc")(fc_full)
        visibility_logits_unmasked = Dense(units=21, name="visibility_logits_unmasked")(visibility_fc)
        visibility_logits = Multiply(name="visibility_logits")([visibility_logits_unmasked, detection_mask])
        pose_fc = Dense(units=512, activation="relu", name="pose_fc")(fc_full)
        pose_logits_unmasked = Dense(units=3, name="pose_logits_unmasked")(pose_fc)
        pose_logits = Multiply(name="pose_logits")([pose_logits_unmasked, detection_mask])
        gender_fc = Dense(units=512, activation="relu", name="gender_fc")(fc_full)
        gender_logits_unmasked = Dense(units=3, name="gender_logits_unmasked")(gender_fc)
        gender_logits = Multiply(name="gender_logits")([gender_logits_unmasked, detection_mask])

        self.model = Model(
            inputs=[input_tensor, visibility_mask, detection_mask],
            output=[face_logits, landmark_logits, visibility_logits, pose_logits, gender_logits],
        )

    def show_model_summary(self):
        if self.model:
            self.model.summary()


class HyperFaceNetworkResNet(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None
        self._create_graph()

    def _create_graph(self):
        raise NotImplementedError("The architecture is not implemented yet")

    def show_model_summary(self):
        if self.model:
            self.model.summary()


if __name__ == "__main__":
    h_face = HyperFaceNetworkAlexNet((227, 227, 3))
    h_face.show_model_summary()
