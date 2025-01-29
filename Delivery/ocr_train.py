from OCR import OCR
import os

'''
    Small script to train the OCR model
    Authors: Rustenis, Julius
'''

root_dir = root_dir = os.path.abspath(os.path.dirname(__file__))
ocr_dir = os.path.join(root_dir, 'OCR')
dataset_dir = os.path.join(ocr_dir, 'Dataset')

ocr = OCR()
ocr.train_model(path_to_training_set=dataset_dir)
ocr.export_model("ReallyFinalModel.mdl")