import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle


class OCR:
    model = OneVsOneClassifier(LinearSVC(random_state=0))
    training_mean = 0
    training_std = 1

    def __init__(self, path_to_model: str = None):
        if path_to_model is None or path_to_model == "":
            print("Using newly inited model")
        elif not self.load_model(path_to_model):
            print("Could not find model at " + path_to_model)

    def load_model(self, path_to_model: str) -> bool:
        """
        Imports a previously exported model and saves it as this objects member.
        :param path_to_model: The relative or absolute path including the filename
        :return: True if operation was successful
        """
        try:
            loaded_data = pickle.load(open(path_to_model, 'rb'))
            self.model = loaded_data["model"]
            self.training_mean = loaded_data["mean"]
            self.training_std = loaded_data["std"]
            return True
        except FileNotFoundError:
            print(f"Error: File '{path_to_model}' not found.")
            return False
        except IsADirectoryError:
            print(f"Error: '{path_to_model}' is a directory, not a file.")
            return False
        except pickle.UnpicklingError:
            print(f"Error: File '{path_to_model}' is not a valid pickle file.")
            return False
        except EOFError:
            print(f"Error: File '{path_to_model}' is empty or incomplete.")
            return False
        except PermissionError:
            print(f"Error: Permission denied to read file '{path_to_model}'.")
            return False
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False

    def export_model(self, path_to_model: str):
        """
        Dumps the current model as a serialized object
        :param path_to_model: The relative or absolute path which also contains the file name
        """
        directory = os.path.dirname(path_to_model)
        if directory and not os.path.exists(directory):
            print(f"Error while exporting model: Directory '{directory}' does not exist.")
            exit(-1)
        data_to_save = {
            "model": self.model,
            "mean": self.training_mean,
            "std": self.training_std
        }
        pickle.dump(data_to_save, open(path_to_model, 'wb'))

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

        # Standardize the data
        self.training_mean = np.mean(features, axis=0)
        self.training_std = np.std(features, axis=0)
        standardized_data = (features - self.training_mean) / self.training_std

        # Split depending on argument
        x_train, x_test, y_train, y_test = train_test_split(standardized_data, labels, test_size=test_size,
                                                            shuffle=True, random_state=69)

        # Train the classifier
        self.model.fit(x_train, y_train)

        # Predict labels for the test set and evaluate
        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")

    def predict(self, image: cv2.typing.MatLike) -> str:
        """
        Extracts features from the given image and uses them to predict the label, given as a character
        :param image: A cv2.typing.MatLike, which is assumed to be black and white and only of a single character
        :return: string which is always a single character
        """
        feature_vector = np.array(self.extract_features(image), dtype=np.float32)
        standardized_data = (feature_vector - self.training_mean) / self.training_std
        y_pred = self.model.predict(standardized_data.reshape(1, -1))
        return y_pred[0]

    def extract_features(self, image: cv2.typing.MatLike) -> [float]:
        """
        Extracts multiple different features if the given image and collects them into a list of numbers
        :param image: A cv2.typing.MatLike, which is assumed to be black and white and only of a single character
        :return:
        """
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
    ocr = OCR("FinalTrained.mdl")
