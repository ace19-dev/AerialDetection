'''
https://dacon.io/competitions/official/235644/codeshre/1698?page=1&dtype=recent&ptype=pub
'''

import os
# from DOTA_devkit.DOTA import DOTA
import DOTA_devkit.dota_utils as util
import pylab

import numpy as np
import matplotlib.pyplot as plt

from arirang import Arirang

pylab.rcParams['figure.figsize'] = (15.0, 15.0)

data_path = 'demo'
train_data = Arirang(data_path)

categories = ['small ship', 'large ship', 'civilian aircraft', 'military aircraft',
              'small car', 'bus', 'truck', 'train', 'crane', 'bridge', 'oil tank',
              'dam', 'athletic field', 'helipad', 'roundabout', 'etc']

# all the image ids contain the categories
# imgids = train_data.getImgIds(catNms=categories)
# imgid = imgids[0]

# specify image index
for imgid in train_data.imglist:
    img = train_data.loadImgs(imgid)[0]

    anns = train_data.loadAnns(imgId=imgid)
    # print(anns)
    result = train_data.showAnns(anns, imgid, 2)
    # plt.show()
    plt.savefig('demo/out/'+imgid+'.png', dpi=150)
    plt.close()

# +++++++++++++++++++++++++++++++++++++++++++++++++

# # Patch 쪼개기
# os.makedirs("examplesplit")
# from ImgSplit import splitbase
#
# split = splitbase(r'example', r'examplesplit', choosebestpoint=True)
# split.splitdata(0.5)
# split.splitdata(1)
# split.splitdata(2)
#
# examplesplit = DOTA('./examplesplit')
# imgids = examplesplit.getImgIds(catNms=['plane'])
# imgid = imgids[1]
# img = examplesplit.loadImgs(imgid)[0]
#
# anns = examplesplit.loadAnns(imgId=imgid)
# # print(anns)
# examplesplit.showAnns(anns, imgid, 2)
#
# plt.axis('off')
# plt.imshow(img)
# plt.show()
