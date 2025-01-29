import re

'''
    Function to analyse the input string and return the plate information
    Author: Abraham
'''
def PlateAnalyseInput(string_input):

    plateData = PlateData()
    plateInput = string_input.strip()
    plateData = plateData.setPlateInfo(plateInput)
    out = plateData.getPlateInfo()
    return out
class PlateData:
    def __init__(self):
        self.fullPlate = ""
        self.classification_ID = 0
        self.classification_string = ""
        self.voivodeship_ID = ''
        self.voivodeship_string = ""
        self.powiat = ""

    def getPlateInfo(self):
        output = (
        f"Plate Number: {self.fullPlate}\n"
        f"Plate Type: {self.classification_string}\n"
        f"Plate Voivodeship: {self.voivodeship_string}"
        )
        return output

    def setPlateInfo(self, plateInput):
        self.fullPlate = self.sanitisePlate(plateInput)
        self.findPlateClassification()
        self.findPlateVoivodeship()
        return self

    def findPlateClassificationID(self):
        if self.fullPlate[0] == 'H':
            return 9
        elif self.fullPlate[0] == 'U':
            return 10
        else:
            if self.fullPlate[0].isdigit():
                return 0

            plateLength = len(self.fullPlate)
            if plateLength == 5:
                return 4
            elif plateLength == 4:
                return 3
            elif plateLength == 8:
                return 1
            elif plateLength == 6:
                if self.fullPlate[1].isdigit():
                    if self.fullPlate[5] == 'B':
                        return 11
                    else:
                        return 5
                else:
                    return 2
            elif plateLength == 7:
                if self.fullPlate[1].isdigit():
                    if self.fullPlate[2].isdigit():
                        return 8
                    else:
                        return 6
                else:
                    return 1
            else:
                return 0

    def findPlateClassification(self):
        self.classification_ID = self.findPlateClassificationID()
        classification_map = {
            0: "Undefined",
            1: "Cars, trucks, and buses",
            2: "Motorcycles, mopeds, and agricultural vehicles",
            3: "Cars – reduced size",
            4: "Classic cars",
            5: "Temporary and export plates",
            6: "Custom plates",
            7: "Professional plates",
            8: "Diplomatic plates",
            9: "Service plates",
            10: "Military plates",
            11: "Testing vehicles"
        }
        self.classification_string = classification_map.get(self.classification_ID, "Undefined")

    def findPlateVoivodeship(self):
        self.voivodeship_ID = self.findPlateVoivodeshipID()
        voivodeship_map = {
            'D': "Lower Silesian Voivodeship",
            'V': "Lower Silesian Voivodeship",
            'C': "Kuyavian-Pomeranian Voivodeship",
            'E': "Łódź Voivodeship",
            'L': "Lublin Voivodeship",
            'F': "Lubusz Voivodeship",
            'K': "Lesser Poland Voivodeship",
            'J': "Lesser Poland Voivodeship",
            'W': "Masovian Voivodeship",
            'A': "Masovian Voivodeship",
            'O': "Opole Voivodeship",
            'R': "Podkarpackie Voivodeship",
            'Y': "Podkarpackie Voivodeship",
            'B': "Podlaskie Voivodeship",
            'G': "Pomeranian Voivodeship",
            'X': "Pomeranian Voivodeship",
            'S': "Silesian Voivodeship",
            'I': "Silesian Voivodeship",
            'T': "Świętokrzyskie Voivodeship",
            'N': "Warmian-Masurian Voivodeship",
            'P': "Greater Poland Voivodeship",
            'M': "Greater Poland Voivodeship",
            'Z': "West Pomeranian Voivodeship"
        }
        self.voivodeship_string = voivodeship_map.get(self.voivodeship_ID, "Unknown")

    def findPlateVoivodeshipID(self):
        return self.fullPlate[0]

    def sanitisePlate(self, plateInput):
        plateOutput = re.sub(r'[^a-zA-Z0-9]', '', plateInput)
        plateOutput = plateOutput[:8]
        plateOutput = plateOutput.upper()
        return plateOutput