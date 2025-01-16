import os
from ROI import RegionOfInterest
from PROC import RectangleToTrapezoid

root_dir = root_dir = os.path.abspath(os.path.dirname(__file__))
test_img = os.path.join(root_dir, 'test.jpg')

roi = RegionOfInterest()
images = roi.detect(test_img)
coords = RectangleToTrapezoid(images[0])