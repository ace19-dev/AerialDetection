import os
import numpy as np
import cv2
import copy
from tqdm import tqdm
import DOTA_devkit.dota_utils as util


class splitbase():
    def __init__(self,
                 srcpath,
                 dstpath,
                 gap=512,
                 # gap=512,
                 subsize=1024,
                 ext='.png'):
        self.srcpath = srcpath
        self.outpath = dstpath
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.srcimagepath = os.path.join(self.srcpath, 'images')
        self.outimagepath = os.path.join(self.outpath, 'images')
        self.ext = ext

        if not os.path.isdir(self.outpath):
            os.mkdir(self.outpath)
        if not os.path.isdir(self.outimagepath):
            os.mkdir(self.outimagepath)

    def save_image_patches(self, img, subimgname, left, up):
        subimg = copy.deepcopy(img[up: (up + self.subsize), left: (left + self.subsize)])
        outdir = os.path.join(self.outimagepath, subimgname + self.ext)
        cv2.imwrite(outdir, subimg)

    def SplitSingle(self, name, rate, extent):
        img = cv2.imread(os.path.join(self.srcimagepath, name + extent), cv2.IMREAD_UNCHANGED)
        assert np.shape(img) != ()

        if (rate != 1):
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
        else:
            resizeimg = img
        outbasename = name + '__' + str(rate) + '__'

        weight = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]

        left, up = 0, 0
        while (left < weight):
            if (left + self.subsize >= weight):
                left = max(weight - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                subimgname = outbasename + str(left) + '___' + str(up)
                self.save_image_patches(resizeimg, subimgname, left, up)
                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= weight):
                break
            else:
                left = left + self.slide

    def splitdata(self, rate):
        imagelist = util.GetFileFromThisRootDir(self.srcpath)
        imagenames = [util.custombasename(x) for x in imagelist if (util.custombasename(x) != 'Thumbs')]
        for name in tqdm(imagenames):
            self.SplitSingle(name, rate, self.ext)


if __name__ == '__main__':
    split = splitbase(r'/home/dingjian/data/GF3Process/tiff',
                      r'/home/dingjian/data/GF3Process/subimg',
                      ext='.tiff')
    split.splitdata(1)
