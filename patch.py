import matplotlib.pyplot as plt

from DOTA_devkit.DOTA import DOTA

###############################
# patch image with annotation
###############################
from DOTA_devkit.ImgSplit import splitbase

src = '/home/ace19/dl_data/Arirang_Dataset/train'
out = '/home/ace19/dl_data/Arirang_Dataset/patch2'

split = splitbase(src, out, choosebestpoint=True)
split.splitdata(0.5)
split.splitdata(1)
split.splitdata(2)
# split.splitdata(3)


#####################################
# patch image without annotation
#####################################
from DOTA_devkit.SplitOnlyImage import splitbase

src = '/home/ace19/dl_data/Arirang_Dataset/final'
out = '/home/ace19/dl_data/Arirang_Dataset/patch_final'

split = splitbase(src, out)
split.splitdata(0.5)
split.splitdata(1)
split.splitdata(2)
# split.splitdata(3)


#######################################################
# examplesplit = DOTA('examplesplit')
# # examplesplit = Arirang('../examplesplit')
# imgids = examplesplit.getImgIds(catNms=['small'])
# imgid = imgids[2]
# img = examplesplit.loadImgs(imgid)[0]
#
# plt.axis('off')
# plt.imshow(img)
# plt.show()
#
# anns = examplesplit.loadAnns(imgId=imgid)
# # print(anns)
# examplesplit.showAnns(anns, imgid, 2)
