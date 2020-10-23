import os
import csv

root = '/home/ace19/my-repo/AerialDetection/submit/results/merged'

_path = os.listdir(root)
_path.sort()

submit_results = []
total = len(_path)
for idx, p in enumerate(_path):
    merged_txt = os.path.join(root, p, 'restored', p + '.txt')
    f = open(merged_txt, 'r')
    lines = f.readlines()
    for line in lines:
        splitlines = line.strip().split(',')
        if splitlines[8] == 'etc':
            continue

        object_struct = {}
        object_struct['file_name'] = p + '.png'
        object_struct['class_id'] = splitlines[9]
        object_struct['confidence'] = splitlines[10]
        object_struct['point1_x'] = splitlines[0]
        object_struct['point1_y'] = splitlines[1]
        object_struct['point2_x'] = splitlines[2]
        object_struct['point2_y'] = splitlines[3]
        object_struct['point3_x'] = splitlines[4]
        object_struct['point3_y'] = splitlines[5]
        object_struct['point4_x'] = splitlines[6]
        object_struct['point4_y'] = splitlines[7]
        submit_results.append(object_struct)

    print('%d/%d completed.. ' % (idx+1, total))

# TODO: fix for merge submit
if not os.path.exists('submit'):
    os.makedirs('submit')

fout = open('submit/%s_submission.csv' % 'season-4',
            'w', encoding='UTF-8', newline='')
writer = csv.writer(fout)
writer.writerow(['file_name', 'class_id', 'confidence', 'point1_x', 'point1_y',
                 'point2_x', 'point2_y', 'point3_x', 'point3_y', 'point4_x', 'point4_y', ])
for r in submit_results:
    writer.writerow([r['file_name'], r['class_id'], r['confidence'],
                     r['point1_x'], r['point1_y'],
                     r['point2_x'], r['point2_y'],
                     r['point3_x'], r['point3_y'],
                     r['point4_x'], r['point4_y']])
fout.close()
