import os
from ROI import RegionOfInterest
from PROC import RectangleToTrapezoid, ProcessImage
import cv2
import numpy as np

root_dir = root_dir = os.path.abspath(os.path.dirname(__file__))
test_img = os.path.join(root_dir, 'test.jpg')

roi = RegionOfInterest()
images = roi.detect(test_img)
coords = RectangleToTrapezoid(images[0])
print(coords)
out = ProcessImage(coords, images[0])
for i in out:
    cv2.imshow('image', i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()