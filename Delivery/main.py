import os
from ROI import RegionOfInterest
from PROC import RectangleToTrapezoid, ProcessImage
from OCR import OCR
import glob

root_dir = root_dir = os.path.abspath(os.path.dirname(__file__))
test_images = glob.glob(os.path.join(root_dir, 'Test_Data', '*.jpg'))

roi = RegionOfInterest()
ocr = OCR("FirstTest.mdl")

for test_img in test_images[0]:

    regions = roi.detect(test_img)
    if not regions:
        print("No license plate detected")
        break

    for r in regions:
        coords = RectangleToTrapezoid(r)
        print(f"Detected license plate coordinates: {coords}")
        out = ProcessImage(coords, r)

        if not out:
            print(f"Error processing image: {test_img}")
            break

        number = []
        for o in out:
            number.append(ocr.predict(o))
        
        print(f"Detected license plate number: {''.join(number)}")
