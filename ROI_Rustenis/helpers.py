import matplotlib.pyplot as plt
import matplotlib.patches as patches
from lxml import etree

def plot_with_bounding_box(img_path, bboxes, bb_format):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    image = plt.imread(img_path)
    plt.imshow(image)
    for _, bbox in enumerate(bboxes):
        if bb_format == "xywh":
            xc, yc, w, h = bbox
            xmin = xc - (w/2)
            ymin = yc - (h/2)
        else:
            xmin, ymin, xmax, ymax = bbox
            w = xmax - xmin
            h = ymax - ymin
        box = patches.Rectangle((xmin, ymin), w, h, edgecolor="red", facecolor="none")
        ax.add_patch(box)
    plt.axis('off')
    plt.savefig('Out/BoundingBoxes.png')

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