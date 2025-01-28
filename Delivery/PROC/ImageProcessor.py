import numpy as np
import cv2
import os

root_dir = os.path.abspath(os.path.dirname(__file__))
char_gen_dir = os.path.join(root_dir, 'Generated_Characters')

'''
    Function to process the image
    Author: Abraham
'''
def ProcessImage(array, image_input, verbose=True):
    output = []
    output_unordered = []

    #Read Image File
    assert image_input is not None, "file could not be read, check with os.path.exists()"

    #Set Coordinate
    ##Currently Manually Set
    xy = [[int(point[0]), int(point[1])] for point in array]

    #Crop Image to Coordinate
    xylength = abs(xy[0][0]-xy[1][0])
    xyheight = abs(xy[0][1]-xy[2][1])
    pts1 = np.float32(xy)
    pts2 = np.float32([[0,0],[xylength,0],[0,xyheight],[xylength,xyheight]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    image_process_crop = cv2.warpPerspective(image_input,M,(xylength,xyheight))

    # Uniformly Resize Image
    standardsize = (500,100)
    image_process_resize = cv2.resize(image_process_crop, standardsize, interpolation= cv2.INTER_LINEAR)

    #Apply Smoothing with Gaussian Blur
    image_process_smooth = cv2.GaussianBlur(image_process_resize, (9, 9), 1)

    #Preprocess Image to GrayScale
    image_process_gray= cv2.cvtColor(image_process_smooth, cv2.COLOR_BGR2GRAY)

    #Turn Image into B/W Binary using Otsu method
    (_, image_process_bw) = cv2.threshold(image_process_gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    #Bounding Box Mask

    _, labels = cv2.connectedComponents(image_process_bw)
    mask = np.zeros(image_process_bw.shape, dtype="uint8")

    # Set lower bound and upper bound criteria for characters
    total_pixels = image_process_bw.shape[0] * image_process_bw.shape[1]
    lower = total_pixels // 100 # heuristic param, can be fine tuned if necessary
    upper = total_pixels // 10 # heuristic param, can be fine tuned if necessary
    upper_width = 100 # Max Width to ~100px

    # Loop over the unique components
    for (_, label) in enumerate(np.unique(labels)):
        # If this is the background label, ignore it
        if label == 0:
            continue

        # Otherwise, construct the label mask to display only connected component
        # for the current label
        labelMask = np.zeros(image_process_bw.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        x, y, w, h = cv2.boundingRect(labelMask)

        # If the number of pixels in the component is between lower bound and upper bound,
        # add it to our mask
        if lower < numPixels < upper and w < upper_width:
            mask = cv2.add(mask, labelMask)
            mask2 = image_process_bw[y:y+h,x:x+w]
            output_unordered.append(mask2)

    #REORDER CHARACTERS
    #function to get the first element
    def takeFirstElm(ele):
        return ele[0]

    #function to order the array using the first element(x-axis)
    def reorder_first_index(list):
        return sorted(list,key=takeFirstElm)

        ordered_elements = reorder_first_index(output_unordered)

        #removing the x-axis from the elements
        output=[]
        for element in ordered_elements:
            output.append(element[1])# appending only the image pixels(removing added index in earlier steps)

            if not output:
                # Provide feedback if no character is found
                # If "Verbose" is set to False, this will be skipped
                if verbose:
                    cv2.imshow('Input Image', image_input)
                    print(f'Cropped with coords {array}')
                    cv2.imshow('Cropped', image_process_crop)
                    cv2.imshow('.preprocess/1_Normalize.jpg', image_process_resize)
                    cv2.imshow('.preprocess/2_Blur.jpg', image_process_smooth)
                    cv2.imshow('.preprocess/3_Grayscale.jpg', image_process_gray)
                    cv2.imshow('.preprocess/4_BlackWhite.jpg', image_process_bw)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            return output

'''
    Function to generate characters from the given images
    Used for OCR training
    Author: Rustenis
'''
def generate_characters(array, image_input, label):
    out = ProcessImage(array, image_input, verbose=False)

    if not out:
        print(f"{label}: No characters found")
        return

    if not os.path.exists(char_gen_dir):
        os.makedirs(char_gen_dir)

    for i, o in enumerate(out):
        cv2.imwrite(os.path.join(char_gen_dir, f'{label}_{i}.jpg'), o)
    
