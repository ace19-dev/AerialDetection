import os

from DOTA_devkit.ResultMerge import mergebypoly
from DOTA_devkit.DOTA import DOTA
import DOTA_devkit.dota_utils as util

root = '/home/ace19/my-repo/AerialDetection/submit/results'

# util.groundtruth2Task1(r'examplesplit/labelTxt', r'Task1')
# mergebypoly(r'Task1', r'Task1_merge')
# util.Task2groundtruth_poly(r'Task1_merge', r'restoredexample/labelTxt')
util.groundtruth2task(os.path.join(root, 'json'),
                       os.path.join(root, 'task'))
mergebypoly(os.path.join(root, 'task'),
            os.path.join(root, 'task_merged'))
util.task2groundtruth_poly(os.path.join(root, 'task_merged'),
                           os.path.join(root, 'restored'))
                           # r'restoredexample/labelTxt')

# filepath = 'example/labelTxt'
# imgids = util.GetFileFromThisRootDir(filepath)
# imgids = [util.custombasename(x) for x in imgids]
# print(imgids)
# ['P0770', 'P1234', 'P1088', 'P2709', 'P0706', 'P1888', 'P2598']

# example = DOTA(r'example')
# num = 2
# anns = example.loadAnns(imgId=imgids[num])
# # print(anns)
# example.showAnns(anns, imgids[num], 2)


# restored = DOTA(r'restoredexample')
# num = 2
# anns = restored.loadAnns(imgId=imgids[num])
# # print(anns)
# restored.showAnns(anns, imgids[num], 2)
