import os
from ROI import RegionOfInterest

root_dir = root_dir = os.path.abspath(os.path.dirname(__file__))
test_img = os.path.join(root_dir, 'test.jpg')

roi = RegionOfInterest()
bboxes = roi.detect(test_img)
print(bboxes)