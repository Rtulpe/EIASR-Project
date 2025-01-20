import cv2
import numpy as np

def RectangleToTrapezoid(image, verbose=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = []
    # Used in case of failure
    debug_area = []

    # Sometimes the biggest contour is only a part of the license plate
    # Thus, merging all the big contours should gives us the entire plate
    for c in contours:
        area = cv2.contourArea(c)
        debug_area.append(area)

        if area > 500:
            large_contours.append(c)

    if large_contours:
        merged = np.vstack(large_contours)
        rect = cv2.minAreaRect(merged)
        box = cv2.boxPoints(rect)
        box = np.array(box).astype(int)
        
        height, width = image.shape[:2]

        box[:, 0] = np.clip(box[:, 0], 0, width - 1)
        box[:, 1] = np.clip(box[:, 1], 0, height - 1)

        results = []

        sorted_points = sorted(box, key=lambda p: (p[1], p[0]))
        top_two = sorted(sorted_points[:2], key=lambda p: p[0])
        bottom_two = sorted(sorted_points[2:], key=lambda p: p[0])

        results.append(top_two[0].tolist())
        results.append(top_two[1].tolist())
        results.append(bottom_two[0].tolist())
        results.append(bottom_two[1].tolist())
        
        return results

    if verbose:
        print("Failed with given areas: {}".format(debug_area))
        cv2.imshow("R2T Image", image)
        cv2.imshow("R2T Binary", binary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return None



