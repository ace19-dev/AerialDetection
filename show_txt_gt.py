import numpy as np
import matplotlib.pyplot as plt
import os
from DOTA_devkit.DOTA import DOTA
# import dota_utils as util
import pylab
pylab.rcParams['figure.figsize'] = (15.0, 15.0)

example = DOTA('demo')

imgids = example.getImgIds()
imgid = imgids[1]
# img = example.loadImgs(imgid)[0]
#
# plt.axis('off')
#
# plt.imshow(img)
# plt.show()

anns = example.loadAnns(imgId=imgid)
# print(anns)
example.showAnns(anns, imgid, 2)
plt.show()