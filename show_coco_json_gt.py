import os
import json
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Circle, Rectangle


def parse_arirang(filename):
    """
        parse the dota ground truth in the format:
        [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    objects = []

    jsons = glob(os.path.join(filename))

    with open(jsons[0]) as f:
        data_dict = json.load(f)
    images = data_dict['images']
    images_name = [img['file_name'] for img in images]
    categories = data_dict['categories']
    categories_name = [cate['name'] for cate in categories]
    annotations = data_dict['annotations']

    for anno in tqdm(annotations, desc='extracting annotations'):
        object_struct = {}

        object_struct['area'] = anno['area']
        object_struct['category_name'] = categories_name[anno['category_id'] - 1]
        # object_struct['segmentation'] = anno['segmentation']
        object_struct['bbox'] = anno['bbox']
        object_struct['image_id'] = anno['image_id']
        object_struct['image_name'] = images_name[anno['image_id'] - 1]

        objects.append(object_struct)

    return objects


def getfile_from_rootdir(dir, ext=None):
    allfiles = []
    needExtFilter = (ext != None)
    for root, dirs, files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles


def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])


def showAnns(imgId, object):
    filename = imgId + '.png'
    img = cv2.imread(filename)

    plt.imshow(img)
    plt.axis('off')

    ax = plt.gca()
    ax.set_autoscale_on(True)
    polygons = []
    color = []
    circles = []

    r = 3
    for obj in object:
        c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        bbox = obj['bbox']
        # xmin, ymin, w, h -> clock wise 8 coords
        p1 = (bbox[0], bbox[1])
        p2 = (bbox[0] + bbox[2], bbox[1])
        p3 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
        p4 = (bbox[0], bbox[1] + bbox[3])
        poly = [p1, p2, p3, p4]
        polygons.append(Polygon(poly))
        color.append(c)
        point = poly[0]
        circle = Circle((point[0], point[1]), r)
        circles.append(circle)

    p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.6)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=1)
    ax.add_collection(p)
    p = PatchCollection(circles, facecolors='red')
    ax.add_collection(p)


def filtering(image_id, objects):
    filtered = []

    image_id = image_id + '.png'
    for obj in objects:
        if obj['image_name'] == image_id:
            filtered.append(obj)

    return filtered


basepath = 'demo'
labelpath = os.path.join(basepath, 'json')
imagepath = os.path.join(basepath, 'images')
imgpaths = getfile_from_rootdir(labelpath)
imglist = [custombasename(x) for x in imgpaths]
imglist.sort()

objects = parse_arirang(os.path.join(basepath, 'demo.json'))

img = imglist[3]
obj = filtering(img, objects)
showAnns(os.path.join(imagepath, img), obj)
plt.show()
