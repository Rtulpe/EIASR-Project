import os
import cv2
import numpy as np
from keras.api.models import load_model

from preproc import preprocess_image

# Define the post-processing function to extract bounding boxes
def postprocess_predictions(predictions, input_size=(416, 416), confidence_threshold=0.5):
    boxes = []
    confidences = []
    class_probs = []

    for i in range(predictions.shape[1]):  # Iterate over the grid cells (height)
        for j in range(predictions.shape[2]):  # Iterate over the grid cells (width)
            # Extract the prediction
            pred = predictions[0, i, j]  # We use [0] since it's a single image batch

            # Extract the center coordinates, width, height, and confidence
            x_center, y_center, width, height, confidence = pred[:5]
            class_prob = pred[5:]

            # Only keep predictions with high confidence
            if confidence >= confidence_threshold:
                # Convert from relative to absolute coordinates
                xmin = int((x_center - width / 2) * input_size[1])
                ymin = int((y_center - height / 2) * input_size[0])
                xmax = int((x_center + width / 2) * input_size[1])
                ymax = int((y_center + height / 2) * input_size[0])

                # Append the bounding box and confidence score
                boxes.append([xmin, ymin, xmax, ymax])
                confidences.append(confidence)
                class_probs.append(class_prob)  # Append the class probabilities
        
    return boxes, confidences, class_probs

# Define the function to get predictions
def get_predictions(model, image, input_size=(416, 416)):
    # Run the image through the model
    predictions = model.predict(np.expand_dims(image, axis=0))
    
    # Post-process the predictions: extract bounding boxes, confidence, and class probabilities
    boxes, confidences, class_probs = postprocess_predictions(predictions, input_size)
    
    return boxes, confidences, class_probs

# Define the function to draw bounding boxes
def draw_bounding_boxes(image_path, boxes, confidences, output_path="output_image.png"):
    # Load the original image
    image = cv2.imread(image_path)
    
    # Draw each bounding box on the image
    for box, confidence in zip(boxes, confidences):
        xmin, ymin, xmax, ymax = box
        
        # Draw a rectangle around the license plate
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Green rectangle
        
        # Add confidence text above the rectangle
        label = f"{confidence:.2f}"
        cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the output image with bounding boxes
    cv2.imwrite(output_path, image)


# Load the trained YOLO model
model = load_model('model.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Main loop to iterate over all images in the Samples folder
samples_folder = "Samples"
output_folder = "Output"

# Create an output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all .jpg images in the Samples folder
for filename in os.listdir(samples_folder):
    print(f"Processing: {filename}")
    if filename.endswith(".png"):
        image_path = os.path.join(samples_folder, filename)
        
        try:
            # Step 1: Preprocess the image
            preprocessed_image = preprocess_image(image_path)

            # Step 2: Get predictions from the model
            boxes, confidences, _ = get_predictions(model, preprocessed_image)

            # Step 3: Draw bounding boxes on the image
            output_path = os.path.join(output_folder, f"output_{filename}")
            print("Drawing bounding boxes...")
            draw_bounding_boxes(image_path, boxes, confidences, output_path)

            print(f"Processed and saved: {output_path}")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
