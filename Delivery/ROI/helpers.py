from lxml import etree
import cv2

def bounding_box_in_yolo_format(x1, y1, x2, y2, w, h):
    x1_new = x1/w
    x2_new = x2/w
    y1_new = y1/h
    y2_new = y2/h
    width_bbox = x2_new - x1_new
    height_bbox = y2_new - y1_new
    x = x1_new + (width_bbox/2)
    y = y1_new + (height_bbox/2)
    return x, y, width_bbox, height_bbox

def process_anns_file(ann_file):
    doc = etree.parse(ann_file)
    x1 = doc.find('.//xmin').text
    y1 = doc.find('.//ymin').text
    x2 = doc.find('.//xmax').text
    y2 = doc.find('.//ymax').text
    w = doc.find('.//width').text
    h = doc.find('.//height').text
    return int(x1),int(y1),int(x2),int(y2),int(w),int(h)

def cut_image(image_path, x1, y1, x2, y2):
    img = cv2.imread(image_path)
    return img[y1:y2, x1:x2]