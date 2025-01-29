'''
    Script to generate characters from the given images
    Used for OCR training
    Author: Rustenis
'''
import os
from ROI import RegionOfInterest
from PROC import RectangleToTrapezoid, generate_characters
import glob

root_dir = root_dir = os.path.abspath(os.path.dirname(__file__))
test_images = glob.glob(os.path.join(root_dir, 'Test_Data', '*.jpg'))

roi = RegionOfInterest()

for test_img in test_images:

    regions = roi.detect(test_img)
    if not regions:
        print(f"{test_img}: Failed ROI detection")
        continue

    for idr, r in enumerate(regions):
        coords = RectangleToTrapezoid(r, verbose=False)

        if not coords:
            print(f"{test_img}: Failed to convert rectangle to trapezoid")
            continue

        
        label = test_img.split('/')[-1].split('.')[0]
        # Have in mind multiple regions can be detected
        # So need to avoid overwriting the images
        label = f"{label}_{idr}"
        generate_characters(coords, r, label)
