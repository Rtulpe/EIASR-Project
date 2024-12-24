def ProcessCSVImageArray(array):
    #Import Packages
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import os

    #Create Output Folder
    #newpath = r'./.preprocess'
    #if not os.path.exists(newpath):
    #    os.makedirs(newpath)
    #Create Output Folder 2
    newpath = r'./charout'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    #Read Image File
    image_input = cv2.imread(array[0])
    assert image_input is not None, "file could not be read, check with os.path.exists()"
    rows,cols,ch = image_input.shape

    #Set Coordinate
    ##Currently Manually Set
    xy  = [[int(array[1]),int(array[2])],[int(array[3]),int(array[4])],[int(array[5]),int(array[6])],[int(array[7]),int(array[8])]]

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
    #cv2.imwrite('.preprocess/1_Normalize.jpg', image_process_resize)

    #Apply Smoothing with Gaussian Blur
    image_process_smooth = cv2.GaussianBlur(image_process_resize, (9, 9), 1)
    #cv2.imwrite('.preprocess/2_Blur.jpg', image_process_smooth)

    #Preprocess Image to GrayScale
    image_process_gray= cv2.cvtColor(image_process_smooth, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite('.preprocess/3_Grayscale.jpg', image_process_gray)

    #Turn Image into B/W Binary using Otsu method
    (thresh, image_process_bw) = cv2.threshold(image_process_gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    #cv2.imwrite('.preprocess/4_BlackWhite.jpg', image_process_bw)



    #Bounding Box Mask

    _, labels = cv2.connectedComponents(image_process_bw)
    mask = np.zeros(image_process_bw.shape, dtype="uint8")

    # Set lower bound and upper bound criteria for characters
    total_pixels = image_process_bw.shape[0] * image_process_bw.shape[1]
    lower = total_pixels // 100 # heuristic param, can be fine tuned if necessary
    upper = total_pixels // 10 # heuristic param, can be fine tuned if necessary
    upper_width = 100 # Max Width to ~100px

    # Loop over the unique components
    for (i, label) in enumerate(np.unique(labels)):
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
            cv2.imwrite('charout/'+array[0]+str(label)+'.jpg', mask2)

    #cv2.imwrite('.preprocess/5_ObjectMask.jpg', mask)


    #Show Bounding Box

    # Get each bounding box
    # Find the big contours/blobs on the filtered image:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    # The Bounding Rectangles will be stored here:
    boundRect = []

    # Look for the outer bounding boxes:
    for i, c in enumerate(contours):

        if hierarchy[0][i][3] == -1:
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect.append(cv2.boundingRect(contours_poly[i]))


    # Draw the bounding boxes on the (copied) input image:
    for i in range(len(boundRect)):
        color = (0, 255, 0)
        cv2.rectangle(image_process_resize, (int(boundRect[i][0]), int(boundRect[i][1])), \
                (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
    #cv2.imwrite('.preprocess/6_OriginalMasked.jpg', image_process_resize)
    #cv2.imwrite('MaskedOutput.jpg', image_process_resize)
