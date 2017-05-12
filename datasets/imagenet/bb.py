'Bounding boxes...'
from config import BOXES_PATH
from tfutils import *
from lxml import etree
from collections import defaultdict
import os
import re


def img_num_from_path(p):
    return int(re.match('.+_(\d+)\.\w+$', p).groups()[0])

def get_coords_from_xml_path(p):
    # returns x0, y0, x1, y1
    s = etree.fromstring(open(p).read())
    bb_by_class = defaultdict(list)
    for obj in s.findall('object'):
        coords = map(lambda x: float(obj.find('bndbox/'+x).text), ('xmin', 'ymin', 'xmax', 'ymax'))
        bb_by_class[obj.find('name').text].append(coords)
    if len(bb_by_class)!=1:
        raise ValueError('Invalid, more than one or no class present! ' + p)
    return bb_by_class.values()[0], (float(s.find('size/width').text), float(s.find('size/height').text))


def get_boxes_by_img_num():
    paths = [os.path.join(BOXES_PATH, p) for p in os.listdir(BOXES_PATH) if p.endswith('.xml')]
    return {img_num_from_path(p):get_coords_from_xml_path(p) for p in paths}