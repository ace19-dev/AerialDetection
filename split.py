'''
 For example, in the splitting process, there are 15 images of patient 1 in the training set,
 and there are 5 images in the validation set; then this may not be ideal since the model has already seen 15
 of the images for patient 1 and can easily remember features that are unique to patient 1, and therefore predict
 well in the validation set for the same patient 1. Therefore, this cross validation method may give
 over optimistic results and fail to generalize well to more unseen images.

 reference on https://www.kaggle.com/reighns/groupkfold-efficientbnet

'''

import argparse
import glob
import os
import random
import shutil
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, StratifiedKFold


def create_kfold(df, num_fold):
    image_file = glob.glob(os.path.join(args.source_dir, 'jpeg/train') + '/*.jpg')
    image_file = [i.split('/')[-1].split('.')[0] for i in image_file]
    print(len(image_file))

    for n in range(num_fold):
        valid = df.loc[df['fold'] == n].index
        train = list(set(image_file) - set(valid))

        print(set(train).intersection(valid))
        np.save(args.target_dir + '/train_fold%d_%d.npy' % (n, len(train)), train)
        np.save(args.target_dir + '/valid_fold%d_%d.npy' % (n, len(valid)), valid)


def _get_stratify_group(row):
    stratify_group = row['sex']
    stratify_group += f'_{row["anatom_site_general_challenge"]}'
    # TODO:
    # stratify_group += f'_{row["source"]}'
    stratify_group += f'_{row["target"]}'

    return stratify_group


def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def split_train_val():
    # df = pd.read_csv(os.path.join(args.source_dir, 'train.csv'))
    # imgid = df['ImageId'].unique().tolist()

    image_file = glob.glob(os.path.join(args.source_dir, 'train') + '/*.jpg')
    image_file = ['train/' + i.split('/')[-1] for i in image_file]
    print(len(image_file))
    # print(image_file[:10])

    # without duplicate
    # duplicate = np.array(DUPLICATE).reshape(-1).tolist()  # 88
    # non_duplicate = list(set(image_file) - set(duplicate))  # 12480
    # random.shuffle(non_duplicate)

    num_fold = 5
    num_valid = int(len(image_file) * 0.1)

    for n in range(num_fold):
        valid = image_file[n * num_valid:(n + 1) * num_valid]
        train = list(set(image_file) - set(valid))

        print(set(train).intersection(valid))
        np.save(args.target_dir + '/train_fold%d_%d.npy' % (n, len(train)), train)
        np.save(args.target_dir + '/valid_fold%d_%d.npy' % (n, len(valid)), valid)


# reference by https://www.kaggle.com/shonenkov/merge-external-data
#
# Provides train/test indices to split data into train/test sets.
# Data are splitted in a way to fulfil the following criteria:
#   - Folds are made by preserving the percentage of samples for each class.
#   - The same group will not appear in two different folds.
def split_train_val2():
    NUM_FOLD = 5

    train_df = pd.read_csv(os.path.join(args.source_dir, 'train.csv'))
    train_df['patient_id'] = train_df['patient_id'].fillna(train_df['image_id'])
    train_df['sex'] = train_df['sex'].fillna('unknown')
    train_df['anatom_site_general_challenge'] = \
        train_df['anatom_site_general_challenge'].fillna('unknown')
    train_df['age_approx'] = train_df['age_approx'].fillna(round(train_df['age_approx'].mean()))
    train_df = train_df.set_index('image_id')

    train_df['stratify_group'] = train_df.apply(_get_stratify_group, axis=1)
    train_df['stratify_group'] = train_df['stratify_group'].astype('category').cat.codes

    train_df.loc[:, 'fold'] = 0

    skf = stratified_group_k_fold(X=train_df.index, y=train_df['stratify_group'],
                                  groups=train_df['patient_id'], k=NUM_FOLD, seed=42)

    for fold_number, (train_index, val_index) in enumerate(skf):
        train_df.loc[train_df.iloc[val_index].index, 'fold'] = fold_number

    train_df.to_csv(os.path.join(args.target_dir, str(NUM_FOLD) + '_folds.csv'))

    create_kfold(train_df, NUM_FOLD)


def split_train_val3():
    total_df = pd.read_csv(os.path.join(args.source_dir, 'train.csv'))
    print('total:\n', total_df.iloc[:, 1:].sum().sort_values(), '\n')

    # https://www.kaggle.com/vishnurapps/undersanding-kfold-stratifiedkfold-and-groupkfold
    NUM_FOLD = 6
    gkf = GroupKFold(n_splits=NUM_FOLD)

    results = []
    for idx, (train_idx, val_idx) in enumerate(gkf.split(total_df, total_df["athletic field"],
                                                         groups=total_df['image name'].tolist())):
        train_fold = total_df.iloc[train_idx]
        val_fold = total_df.iloc[val_idx]

        print('\nsplit: ', idx)
        print('train_df:\n', train_fold.iloc[:, 1:].sum().sort_values())
        print('\nval_df:\n', val_fold.iloc[:, 1:].sum().sort_values())
        print("+++++++++++++++++++++++++++++++\n")

        results.append((train_fold, val_fold))

    for i, splited in enumerate(results):
        train_img = os.path.join(args.target_dir, 'train_fold' + str(i), 'images')
        train_json = os.path.join(args.target_dir, 'train_fold' + str(i), 'json')
        val_img = os.path.join(args.target_dir, 'val_fold' + str(i), 'images')
        val_json = os.path.join(args.target_dir, 'val_fold' + str(i), 'json')

        if not os.path.exists(train_img):
            os.makedirs(train_img)
        if not os.path.exists(train_json):
            os.makedirs(train_json)
        if not os.path.exists(val_img):
            os.makedirs(val_img)
        if not os.path.exists(val_json):
            os.makedirs(val_json)

        for idx, img in enumerate(splited[0]['image name']):
            shutil.copyfile(os.path.join(args.source_dir, 'images', img),
                            os.path.join(train_img, img))
            shutil.copyfile(os.path.join(args.source_dir, 'json', img.split('.')[0] + '.json'),
                            os.path.join(train_json, img.split('.')[0] + '.json'))
        for idx, img in enumerate(splited[1]['image name']):
            shutil.copyfile(os.path.join(args.source_dir, 'images', img),
                            os.path.join(val_img, img))
            shutil.copyfile(os.path.join(args.source_dir, 'json', img.split('.')[0] + '.json'),
                            os.path.join(val_json, img.split('.')[0] + '.json'))

        # train = splited[0]['image_id'].values.tolist()
        # valid = splited[1]['image_id'].values.tolist()
        # np.save(args.target_dir + '/train_fold%d_%d.npy' % (i, len(train)), train)
        # np.save(args.target_dir + '/valid_fold%d_%d.npy' % (i, len(valid)), valid)


'''
each set contains approximately the same percentage of samples of each target class as the complete set.
the test split has at least one y which has value 1

    [ 1  2  3  6  7  8  9 10 12 13 14 15] [ 0  4  5 11]
    [ 0  2  3  4  5  6  7 10 11 12 13 15] [ 1  8  9 14]
    [ 0  1  3  4  5  7  8  9 11 12 13 14] [ 2  6 10 15]
    [ 0  1  2  4  5  6  8  9 10 11 14 15] [ 3  7 12 13]
'''


def test_stratifiedkfold():
    X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    skf = StratifiedKFold(n_splits=4, shuffle=True)
    for train_index, test_index in skf.split(X, y):
        print("%s %s" % (train_index, test_index))


def test_groupkfold():
    X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    groups = ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'd', 'd', 'd', 'd', 'd', 'd']
    gkf = GroupKFold(n_splits=3)
    for train_index, test_index in gkf.split(X, y, groups=groups):
        print("%s %s" % (train_index, test_index))


def test2():
    df = pd.read_csv(os.path.join(args.source_dir, 'sample_submission.csv'))
    # df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
    uid = df.index

    test = ['test/' + i for i in uid]

    # for unsupervsied
    random.shuffle(test)
    num_valid = int(len(uid) * 0.25)
    valid = test[:num_valid]
    train = test[num_valid:]

    np.save(args.target_dir + '/test_train_%d.npy' % len(valid), valid)
    np.save(args.target_dir + '/test_valid_%d.npy' % len(train), train)


def main(args):
    if args.target_dir is not None:
        if not os.path.exists(args.target_dir):
            os.makedirs(args.target_dir)

    # https://www.kaggle.com/vishnurapps/undersanding-kfold-stratifiedkfold-and-groupkfold
    # test_groupkfold()

    # reference on https://www.kaggle.com/reighns/groupkfold-efficientbnet
    split_train_val3()

    # reference on https://www.kaggle.com/shonenkov/merge-external-data
    # split_train_val2()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir',
                        type=str,
                        default='/home/ace19/dl_data/Arirang_Dataset/ori_train',
                        help='Where is train image to load')
    parser.add_argument('--target-dir', type=str,
                        default='/home/ace19/dl_data/Arirang_Dataset',
                        help='Directory to save splited dataset')

    args = parser.parse_args()
    main(args)
