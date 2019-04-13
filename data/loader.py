import numpy
import sys
import cv2
from keras.utils import to_categorical


class DataLoader(object):
    def __init__(self, sample_data_file):
        self.data_file_path = sample_data_file
        self.data = None
        self.train_size = 300
        self._load_in_memory()
        self._prepare_data()

    def _load_in_memory(self):
        self.data = numpy.load(self.data_file_path)

    def print_data_shapes(self):
        print(self.X.shape)
        for l in self.labels:
            print(l[3], l.shape)

    def rescale_and_normalize_images(self, X):
        return numpy.array([cv2.resize(img, (227, 227)) / 255 for img in X])

    def expand_visibility(self, Y, rep):
        expanded_visibility = numpy.tile(numpy.expand_dims(Y, axis=2), [1, 1, rep])
        expanded_visibility = numpy.reshape(expanded_visibility, (expanded_visibility.shape[0], -1))
        return expanded_visibility

    def _mask_label(self, Y, Y_mask, rep):
        expanded_visibility = self.expand_visibility(Y_mask, rep)
        Y_landmark_masked = Y * expanded_visibility
        print(Y_landmark_masked[0])
        return Y_landmark_masked

    def _split_and_transform(self, X, Y_face, Y_landmark, Y_visibility, Y_pose, Y_gender):
        X = numpy.squeeze(X)
        X = self.rescale_and_normalize_images(X)
        Y_face = numpy.array([[face[0][0]] for face in Y_face])
        Y_landmark = numpy.array([landmark[0] for landmark in Y_landmark])
        Y_visibility = numpy.array([visibility[0] for visibility in Y_visibility])
        Y_visibility = Y_face * Y_visibility
        Y_landmark = self._mask_label(Y_landmark, Y_visibility, 2)
        Y_pose = Y_face * numpy.array([pose[0] for pose in Y_pose])
        Y_gender = to_categorical(Y_face * numpy.array([[1] if gender[0][0] == 0 else [2] for gender in Y_gender]))
        # print(Y_gender)
        return X, Y_face, Y_landmark, Y_visibility, Y_pose, Y_gender

    def _prepare_data(self):
        print(self.data.shape)
        X, Y_face, Y_landmark, Y_visibility, Y_pose, Y_gender = numpy.split(self.data, 6, axis=1)
        X, Y_face, Y_landmark, Y_visibility, Y_pose, Y_gender = self._split_and_transform(
            X, Y_face, Y_landmark, Y_visibility, Y_pose, Y_gender
        )
        self.X = X
        self.labels = [Y_face, Y_landmark, Y_visibility, Y_pose, Y_gender]
        # print("Finding non-face")
        # print(Y_face[:,0].shape)


if __name__ == "__main__":
    data_file_path = sys.argv[1]
    data_loader = DataLoader(sample_data_file=data_file_path)
    data_loader.print_data_shapes()
