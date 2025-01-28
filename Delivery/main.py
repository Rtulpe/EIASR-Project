import os
from ROI import RegionOfInterest
from PROC import RectangleToTrapezoid, ProcessImage, PlateAnalyseInput
from OCR import OCR
import glob

root_dir = root_dir = os.path.abspath(os.path.dirname(__file__))
test_images = glob.glob(os.path.join(root_dir, 'Test_Data', '*.jpg'))

roi = RegionOfInterest()
ocr = OCR("ReallyFinalModel.mdl")

for test_img in test_images:

    regions = roi.detect(test_img)
    if not regions:
        print(f"{test_img}: Failed ROI detection")
        continue

    for r in regions:
        coords = RectangleToTrapezoid(r)

        if not coords:
            print(f"{test_img}: Failed to convert rectangle to trapezoid")
            continue

        out = ProcessImage(coords, r)

        if not out:
            print(f"{test_img}: Failed to process image")
            continue

        number = []
        for o in out:
            number.append(ocr.predict(o))

        fullPlate = ''.join(number)
        
        print(f"Detected license plate number: {fullPlate} for image: {test_img}")

        plateInfo=PlateAnalyseInput(fullPlate)
        print(plateInfo)
