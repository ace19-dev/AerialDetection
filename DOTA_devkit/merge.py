import os

from DOTA_devkit.ResultMerge import mergebypoly
from DOTA_devkit.DOTA import DOTA
import DOTA_devkit.dota_utils as util

root = '/home/ace19/my-repo/AerialDetection/submit/results'

path = os.path.join(root, 'json')
json_dirs = os.listdir(path)
json_dirs.sort()

total = len(json_dirs)
for idx, jd in enumerate(json_dirs):
    json_path = os.path.join(path, jd)

    # util.groundtruth2Task1(r'examplesplit/labelTxt', r'Task1')
    # mergebypoly(r'Task1', r'Task1_merge')
    # util.Task2groundtruth_poly(r'Task1_merge', r'restoredexample/labelTxt')
    util.groundtruth2task(json_path, os.path.join(json_path, 'task'))
    mergebypoly(os.path.join(json_path, 'task'),
                os.path.join(json_path, 'task_merged'))
    util.task2groundtruth_poly(os.path.join(json_path, 'task_merged'),
                               os.path.join(json_path, 'restored'))
                               # r'restoredexample/labelTxt')
    print('%d/%d merged.. ' % (idx+1, total))


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
