import os
import json
import glob
import pandas as pd

root_dir = '/home/ace19/dl_data/Arirang_Dataset'

print('학습용 데이터 개수:', len(glob.glob(root_dir + '/ori_train/images/*')))
print('학습용 라벨 개수:', len(glob.glob(root_dir + '/ori_train/json/*')))
print('시험용 데이터 개수:', len(glob.glob(root_dir + '/test/images/*')))

# train label 경로 저장
train_labels = glob.glob(root_dir + '/ori_train/json/*')


def readJSON(path):
    '''
    Parameters:
    ----------
    path: path to json file

    Returns:
    -------
    dict
        dict type of the json file
    '''
    with open(path) as f:
        data = json.load(f)

    return data


def getTypeName(dic):
    '''
    Parameters:
    -----------
    dic: result of readJSON

    Returns:
    --------
    pd.DataFrame
        dataframe with image_id as index and number of each objects contained in the image as values
    '''

    output = {}
    image_id = dic['features'][0]['properties']['image_id']

    for i in range(len(dic['features'])):

        type_name = dic['features'][i]['properties']['type_name']

        if output.get(type_name) == None:
            output[type_name] = 1

        else:
            output[type_name] += 1

    return pd.DataFrame(output, index=[image_id])


total_df = pd.DataFrame()

for path in train_labels:
    total_df = pd.concat([total_df, getTypeName(readJSON(path))], axis=0, join='outer')

total_df.fillna(0, inplace=True)
total_df.index.name = 'image name'

total_df.to_csv(os.path.join(root_dir, 'ori_train', 'train.csv'))
