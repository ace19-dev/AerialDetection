import os
import cv2
import sys
import codecs
import json
import shapely.geometry as shgeo
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Circle
import DOTA_devkit.dota_utils as util
from collections import defaultdict


def _isArrayLike(obj):
    if type(obj) == str:
        return False
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class Arirang:
    def __init__(self, basepath):
        self.basepath = basepath
        self.labelpath = os.path.join(basepath, 'json')
        self.imagepath = os.path.join(basepath, 'images')
        self.labelpaths = getFileFromThisRootDir(self.labelpath)
        self.imglist = [custombasename(x) for x in self.labelpaths]
        self.catToImgs = defaultdict(list)
        self.ImgToAnns = defaultdict(list)
        self.createIndex()

    def createIndex(self):
        for filename in self.labelpaths:
            objects = parse_dota_poly(filename)
            imgid = custombasename(filename)
            self.ImgToAnns[imgid] = objects
            for obj in objects:
                cat = obj['type_name']
                self.catToImgs[cat].append(imgid)

    def getImgIds(self, catNms=[]):
        """
        :param catNms: category names
        :return: all the image ids contain the categories
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        if len(catNms) == 0:
            return self.imglist
        else:
            imgids = []
            for i, cat in enumerate(catNms):
                if i == 0:
                    imgids = set(self.catToImgs[cat])
                else:
                    imgids &= set(self.catToImgs[cat])
        return list(imgids)

    def loadAnns(self, catNms=[], imgId=None, difficult=None):
        """
        :param catNms: category names
        :param imgId: the img to load anns
        :return: objects
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        objects = self.ImgToAnns[imgId]
        if len(catNms) == 0:
            return objects
        outobjects = [obj for obj in objects if (obj['name'] in catNms)]
        return outobjects

    def showAnns(self, objects, imgId, range):
        """
        :param catNms: category names
        :param objects: objects to show
        :param imgId: img to show
        :param range: display range in the img
        :return:
        """
        img = self.loadImgs(imgId)[0]
        plt.imshow(img)
        plt.axis('off')

        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        circles = []
        r = 5
        for obj in objects:
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            poly = obj['poly']
            polygons.append(Polygon(poly))
            color.append(c)
            point = poly[0]
            circle = Circle((point[0], point[1]), r)
            circles.append(circle)
        p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)
        p = PatchCollection(circles, facecolors='red')
        ax.add_collection(p)

    def loadImgs(self, imgids=[]):
        """
        :param imgids: integer ids specifying img
        :return: loaded img objects
        """
        print('isarralike:', _isArrayLike(imgids))
        imgids = imgids if _isArrayLike(imgids) else [imgids]
        print('imgids:', imgids)
        imgs = []
        for imgid in imgids:
            filename = os.path.join(self.imagepath, imgid + '.png')
            print('filename:', filename)
            img = cv2.imread(filename)
            imgs.append(img)
        return imgs


def parse_dota_poly(filename):
    """
        parse the dota ground truth in the format:
        [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    f = []
    if (sys.version_info >= (3, 5)):
        fd = open(filename, 'r')
        f = fd
    elif (sys.version_info >= 2.7):
        fd = codecs.open(filename, 'r')
        f = fd

    with open(filename) as json_file:
        json_data = json.load(json_file)
    features = json_data['features']

    objects = []
    for f in features:
        object_struct = {}
        object_struct['angle'] = f['properties']['object_angle']
        object_struct['image_id'] = f['properties']['image_id']
        object_struct['type_id'] = f['properties']['type_id']
        object_struct['type_name'] = f['properties']['type_name']
        imcoords = f['properties']['object_imcoords'].split(',')
        object_struct['poly'] = [(float(imcoords[0]), float(imcoords[1])),
                                (float(imcoords[2]), float(imcoords[3])),
                                (float(imcoords[4]), float(imcoords[5])),
                                (float(imcoords[6]), float(imcoords[7]))
                                ]
        gtpoly = shgeo.Polygon(object_struct['poly'])
        object_struct['area'] = gtpoly.area
        objects.append(object_struct)

    return objects


def getFileFromThisRootDir(dir, ext=None):
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
