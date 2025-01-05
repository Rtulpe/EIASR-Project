import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

usingBinary = True
debug = False

def process_image(image_path):
    """
    Reads in the image and does multiple processing steps to extract features
    :param image_path: 
    :return: feature vector 
    """""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if usingBinary:
        image = cv2.resize(image, (50, 90))
    else:
        image = cv2.resize(image, (25, 45))  # TODO: Check if this works correctly
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

        # Maybe add projects on the axes and distance from bottom/sides to black pixels

        if debug:
            print(f"{image_path}: Got area {normalized_area}, perimeter {perimeter}, solidity {solidity}, "
                  f"and complexity {shape_complexity}")
        return ([normalized_area, perimeter, solidity, shape_complexity]
                + hu_moments.tolist() + horizontal_projection.tolist() + vertical_projection.tolist())


def train_knn(knn):
    """
    Reads in all the data from a training set and extracts the features. Then it splits it into train and test data
    :param knn: the KNN to be trained
    :return: nothing
    """
    # Define the path to your dataset
    if usingBinary:
        dataset_path = "Dataset/PreBinary/"
    else:
        dataset_path = "InitialTrainingSet/"
    # dataset_path = "Dataset/Manual/"

    # Initialize lists to store features and labels
    features = []
    labels = []

    # Loop through each folder (label) in the dataset
    for label in os.listdir(dataset_path):
        label_specific_features_list = []
        label_specific_name_list = []
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for image_name in os.listdir(label_path):
                if image_name.__contains__(".DS_Store"):  # Needed for Mac-os weirdness
                    continue
                image_path = os.path.join(label_path, image_name)

                # Append the feature vector and label
                feature_vector = process_image(image_path)
                features.append(feature_vector)
                labels.append(label)

                label_specific_features_list.append(feature_vector)
                label_specific_name_list.append(image_name)
        np_features = np.array(label_specific_features_list, dtype=np.float32)
        if debug:
            print(label)
            print(np_features.shape)
        means = np_features.mean(axis=0)
        variances = np_features.var(axis=0)
        # Calculate bounds
        lower_bound = means - 3 * variances
        upper_bound = means + 3 * variances

        # Check if each element in data is within bounds
        boolean_array = (np_features >= lower_bound) & (np_features <= upper_bound)
        if debug:
            for i in range(1, boolean_array.shape[0]):
                print(label_specific_name_list[i])
                print(boolean_array[i])

    # TODO: For KNN features need to be normalized/standardized (mean=0, sigma=1)

    # Convert to NumPy arrays for compatibility with scikit-learn
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True, random_state=42)

    # Train the classifier
    knn.fit(x_train, y_train)

    # Predict labels for the test set
    y_pred = knn.predict(x_test)

    # Evaluate the classifier
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")


def test_knn(knn):
    """
    Predicts the label of some hand selected images and prints if it was correct
    :param knn: a trained KNN
    :return: nothing
    """
    test_data_path = "VerificationSet"
    for label in os.listdir(test_data_path):
        if label.__contains__(".DS_Store"):  # Needed for Mac-os weirdness
            continue
        image_path = os.path.join(test_data_path, label)
        feature_vector = np.array([process_image(image_path)], dtype=np.float32)
        # Predict the label
        predicted_label = knn.predict(feature_vector)
        print(f"Should: {label} - predicted: {predicted_label[0]}")


def main():
    # knn = KNeighborsClassifier(n_neighbors=3)  # Maybe replace with One v One Classifier
    knn = OneVsOneClassifier(LinearSVC(random_state=0))
    train_knn(knn)
    test_knn(knn)


main()
