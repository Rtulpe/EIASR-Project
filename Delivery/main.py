from ROI import RegionOfInterest
from PROC import RectangleToTrapezoid, ProcessImage, PlateAnalyseInput
from OCR import OCR
import argparse

'''
    Main script to run the license plate detection, recognition and identification pipeline
    Authors: Rustenis, Julius, Abraham
'''
def main(image_path):
    roi = RegionOfInterest()
    ocr = OCR("ReallyFinalModel.mdl")

    regions = roi.detect(image_path)
    if not regions:
        print(f"{image_path}: Failed ROI detection")
        return

    for r in regions:
        coords = RectangleToTrapezoid(r)

        if not coords:
            print(f"{image_path}: Failed to convert rectangle to trapezoid")
            continue

        out = ProcessImage(coords, r)

        if not out:
            print(f"{image_path}: Failed to process image")
            continue

        number = []
        for o in out:
            number.append(ocr.predict(o))

        fullPlate = ''.join(number)
        
        print(f"Detected license plate number: {fullPlate} for image: {image_path}")

        plateInfo=PlateAnalyseInput(fullPlate)
        print(plateInfo)


# Argument parsing directly in the script
parser = argparse.ArgumentParser(description="Run main with an image path.")
parser.add_argument("image_path", type=str, help="Path to the image file")
args = parser.parse_args()

# Call main with the image path passed as argument
main(args.image_path)