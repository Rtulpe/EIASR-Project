import cv2
import numpy as np

def RectangleToTrapezoid(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = []

    # Sometimes the biggest contour is only a part of the license plate
    # Thus, merging all the big contours should gives us the entire plate
    for c in contours:
        area = cv2.contourArea(c)
        if area > 2000:
            large_contours.append(c)

    if large_contours:
        merged = np.vstack(large_contours)
        rect = cv2.minAreaRect(merged)
        box = cv2.boxPoints(rect)
        box = np.array(box).astype(int)
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
        # get 4 point coordinates as normal integers
        x1, y1 = box[0]
        x2, y2 = box[1]
        x3, y3 = box[2]
        x4, y4 = box[3]

        return (x1, y1), (x2, y2), (x3, y3), (x4, y4)


    return None
    



