#include <iostream>
#include <regex>
using std::cout;
using std::cin;
using std::endl;


/*
    Function to process the plate and output relevant info
    Author: Abraham
*/
class PlateData {
    public:
        std::string fullPlate;
        int classification_ID = 0;
        std::string classification_string;
        char voivodeship_ID;
        std::string voivodeship_string;
        std::string powiat;

        void printPlateInfo(){
            cout<<"Plate Number: " << fullPlate << endl;
            cout<<"Plate Type: " << classification_string << endl;
            cout<<"Plate Voivodeship: " << voivodeship_string << endl;
        }

        PlateData setPlateInfo(std::string plateInput){
            PlateData plateData;
            plateData.fullPlate = sanitisePlate(plateInput);
            plateData = findPlateClassification(plateData);
            plateData = findPlateVoivodeship(plateData);

            return plateData;
        }
    private:
        int findPlateClassificationID(PlateData plateData){ //0 Undefined;1 Car; 2 Motorcycle; 3 Small Car; 4 Classic Car; 5 TemporaryExport; 6 Custom; 7 Pro; 8 Diplomatic; 9 Service; 10 Military; 11 Testing Vehicles
            if (plateData.fullPlate[0]=='H'){
                return 9;
            } else if (plateData.fullPlate[0]=='U'){
                return 10;
            } else{
                switch (plateData.fullPlate[0]){
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                case '8':
                case '9':
                case '0':
                    return 0;        
                default:
                    break;
                }

                switch (plateData.fullPlate.length()){
                case 5:
                    return 4;
                    break;
                case 4:
                    return 3;
                    break;
                case 8: //Technically there should also be professional plates, but I can't find the proper ruling
                    //return 7;
                    return 1;
                    break;
                case 6:
                    switch (plateData.fullPlate[1]){
                        case '1':
                        case '2':
                        case '3':
                        case '4':
                        case '5':
                        case '6':
                        case '7':
                        case '8':
                        case '9':
                        case '0':
                            switch (plateData.fullPlate[5]){
                            case 'B':
                                return 11;
                                break;

                            default:
                                return 5;
                                break;
                            }
                            break;
                        default:
                            return 2;
                            break;
                    }
                case 7:
                    switch (plateData.fullPlate[1])
                    {
                    case '1':
                    case '2':
                    case '3':
                    case '4':
                    case '5':
                    case '6':
                    case '7':
                    case '8':
                    case '9':
                    case '0':
                        switch (plateData.fullPlate[2]){
                        case '1':
                        case '2':
                        case '3':
                        case '4':
                        case '5':
                        case '6':
                        case '7':
                        case '8':
                        case '9':
                        case '0':
                            return 8;
                            break;
                        
                        default:
                            return 6;
                            break;
                        }
                        break;
                    
                    default: //This may include motorcycles, but there are no differentiation method available
                        return 1;
                        break;
                    }
                default:
                    return 0;
                    break;
                }
            }
        }

        PlateData findPlateClassification(PlateData plateData){
            plateData.classification_ID = findPlateClassificationID(plateData);
            std::string classification_string;
            switch (plateData.classification_ID)
            {
            case 0:
                classification_string = "Undefined";
                break;
            case 1:
                classification_string = "Cars, trucks, and buses";
                break;
            case 2:
                classification_string = "Motorcycles, mopeds, and agricultural vehicles";
                break;
            case 3:
                classification_string = "Cars – reduced size";
                break;
            case 4:
                classification_string = "Classic cars";
                break;
            case 5:
                classification_string = "Temporary and export plates";
                break;
            case 6:
                classification_string = "Custom plates";
                break;
            case 7:
                classification_string = "Professional plates";
                break;
            case 8:
                classification_string = "Diplomatic plates";
                break;
            case 9:
                classification_string = "Service plates";
                break;
            case 10:
                classification_string = "Military plates";
                break;
            case 11:
                classification_string = "Testing vehicles";
                break;
            
            default:
                classification_string = "Undefined";
                break;
            }
            plateData.classification_string = classification_string;
            return plateData;


        }

        PlateData findPlateVoivodeship(PlateData plateData){
            plateData.voivodeship_ID = findPlateVoivodeshipID(plateData);
            std::string voivodeship_string;
            switch (plateData.voivodeship_ID)
            {
            case 'D':
            case 'V':
                voivodeship_string = "Lower Silesian Voivodeship";
                break;
            case 'C':
                voivodeship_string = "Kuyavian-Pomeranian Voivodeship";
                break;
            case 'E':
                voivodeship_string = "Łódź Voivodeship";
                break;
            case 'L':
                voivodeship_string = "Lublin Voivodeship";
                break;
            case 'F':
                voivodeship_string = "Lubusz Voivodeship";
                break;
            case 'K':
            case 'J':
                voivodeship_string = "Lesser Poland Voivodeship";
                break;
            case 'W':
            case 'A':
                voivodeship_string = "Masovian Voivodeship";
                break;
            case 'O':
                voivodeship_string = "Opole Voivodeship";
                break;
            case 'R':
            case 'Y':
                voivodeship_string = "Podkarpackie Voivodeship";
                break;
            case 'B':
                voivodeship_string = "Podlaskie Voivodeship";
                break;
            case 'G':
            case 'X':
                voivodeship_string = "Pomeranian Voivodeship";
                break;
            case 'S':
            case 'I':
                voivodeship_string = "Silesian Voivodeship";
                break;
            case 'T':
                voivodeship_string = "Świętokrzyskie Voivodeship";
                break;
            case 'N':
                voivodeship_string = "Warmian-Masurian Voivodeship";
                break;
            case 'P':
            case 'M':
                voivodeship_string = "Greater Poland Voivodeship";
                break;
            case 'Z':
                voivodeship_string = "West Pomeranian Voivodeship";
                break;
            
            default:
                voivodeship_string = "Unknown";
                break;
            }
            plateData.voivodeship_string = voivodeship_string;
            return plateData;


        }
        char findPlateVoivodeshipID(PlateData plateData){
            return plateData.fullPlate[0];
        }

        std::string sanitisePlate(std::string plateInput){
            std::regex regexCleanString("[^a-zA-Z0-9]");
            std::string plateOutput = std::regex_replace(plateInput, regexCleanString, "");
            plateOutput= plateOutput.substr(0, 8);
            std::transform(plateOutput.begin(), plateOutput.end(), plateOutput.begin(), ::toupper);
            return plateOutput;
        }

};
void printPlateInfo(PlateData plateData);

int main(){
    PlateData plateData;
    std::string plateInput;
    getline(cin, plateInput);
    plateData = plateData.setPlateInfo(plateInput);
    plateData.printPlateInfo();

}




