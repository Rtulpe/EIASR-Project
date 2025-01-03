import unittest

class testPreproc(unittest.TestCase):
    from preproc import parse_annotations
    TRAIN_ANN_DIR = 'Tests'
    TRAIN_IMG_DIR = 'Tests'

    annotations = parse_annotations(TRAIN_ANN_DIR, TRAIN_IMG_DIR)


    def test_parse_annotations(self):
        print(self.annotations)

        self.assertEqual(True, True)

    
    def test_convert_to_yolo_format(self):
        from preproc import convert_to_yolo_format

        TRAIN_ANN_DIR = 'Tests'
        TRAIN_IMG_DIR = 'Tests'

        convert_to_yolo_format(self.annotations, 'Tests', (416, 416))

        self.assertEqual(True, True)