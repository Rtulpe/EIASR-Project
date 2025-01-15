import cv2
import os
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


class OCR:
    # model = KNeighborsClassifier(n_neighbors=3) # Got better results with OneVsOne
    model = OneVsOneClassifier(LinearSVC(random_state=0))

    def __init__(self, path_to_model: str):
        if path_to_model is None or path_to_model == "":
            print("Using newly inited model")
        elif not self.load_model(path_to_model):
            print("Could not find model at " + path_to_model)

    def load_model(self, path_to_model: str) -> bool:
        print("Implement me")

    def export_model(self, path_to_model: str):
        print("Implement me")

    def train_model(self, path_to_training_set: str, test_size=0.2):
        """
        Uses the data in the training set to train the classifier model of this object, but does export it
        :param path_to_training_set: The absolute or relative path to the top of the training data
        :param test_size: Percentage of the training set to be used for determining accuracy
        """
        # Initialize lists to store features and labels
        features = []
        labels = []

        # Loop through each folder (label) in the training set
        for label in os.listdir(path_to_training_set):
            label_path = os.path.join(path_to_training_set, label)
            if os.path.isdir(label_path):
                for image_name in os.listdir(label_path):
                    if image_name.__contains__(".DS_Store"):  # Needed for Mac-os weirdness
                        continue
                    image_path = os.path.join(label_path, image_name)

                    # Append the feature vector and label
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    feature_vector = self.extract_features(image)
                    features.append(feature_vector)
                    labels.append(label)

        # Convert into NumPy arrays for compatibility with scikit-learn
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels)

        # TODO: Features need to be normalized/standardized (mean=0, sigma=1)

        # Split depending on argument # TODO: Check what happens, if it is zero
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size,
                                                            shuffle=True, random_state=42)

        # Train the classifier
        self.model.fit(x_train, y_train)

        # Predict labels for the test set and evaluate
        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")

    def predict(self, image: cv2.typing.MatLike) -> str:
        feature_vector = np.array(self.extract_features(image), dtype=np.float32)
        y_pred = self.model.predict(feature_vector.reshape(1, -1))
        return y_pred[0]

    def extract_features(self, image: cv2.typing.MatLike) -> [float]:
        image = cv2.resize(image, (50, 90))
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)  # also turn into binary edges
        edges = cv2.Canny(image, 100, 200)  # Find edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # only external contours
        if contours:  # There might not be any
            contour = max(contours, key=cv2.contourArea)  # Assume the largest contour corresponds to the letter/number
            # TODO: Check assumption for things like W

            # Compute features
            area = cv2.contourArea(contour)
            normalized_area = area / (image.shape[0] * image.shape[1])
            perimeter = cv2.arcLength(contour, True)
            shape_complexity = (perimeter ** 2) / area if area != 0 else 0
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area != 0 else 0
            moments = cv2.moments(contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            horizontal_projection = np.sum(binary == 0, axis=1)  # Sum of black pixels in each row
            vertical_projection = np.sum(binary == 0, axis=0)  # Sum of black pixels in each column
            return ([normalized_area, perimeter, solidity, shape_complexity]
                    + hu_moments.tolist() + horizontal_projection.tolist() + vertical_projection.tolist())


def main():
    ocr = OCR("")
    ocr.train_model("Dataset/PreBinary/")
    test_image = cv2.imread("VerificationSet/3.jpg", cv2.IMREAD_GRAYSCALE)
    print(ocr.predict(test_image))

main()
