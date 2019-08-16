# import modules
import random
import argparse
import numpy as np
import pandas as pd
from os.path import dirname, realpath, join
from IPython.terminal.debugger import set_trace as keyboard


def main():
    parser = argparse.ArgumentParser(description='generate datasets')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed dataset generation')
    args = parser.parse_args()

    print('Initialize environment')
    # seed the environment
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    # list of levels
    levels = ['level1', 'level2', 'level3', 'level4']
    samples = {'level1': 25, 'level2': 10, 'level3': 5, 'level4': 5}
    thresholds = {'level1': 500, 'level2': 100, 'level3': 50, 'level4': 20}

    # obtain root dir
    root_dir = dirname(realpath(__file__))
    data_path = join(root_dir, 'data')

    # load csv file
    print('Load dataset')
    with open(join(data_path, 'amazon.csv'), 'r', encoding='latin1') as f:
        amazon_df = pd.read_csv(f)

    # rename unnamed column and set as index
    print('Process dataset')
    amazon_df.rename(columns={'Unnamed: 0': 'pid'}, inplace=True)
    amazon_df.set_index('pid', inplace=True)

    # remove rows which have nan
    amazon_df.dropna(axis=0, how='any', subset=['title', 'description'], inplace=True)

    # groupby level labels
    print('Perform grouping')
    groupings = {lvl: amazon_df.groupby(lvl) for lvl in levels}
    groups = {lvl: list(groupings[lvl].groups.keys()) for lvl in levels}
    bools = {lvl: groupings[lvl].count()['title'] < thresholds[lvl] for lvl in levels}
    invalid_groups = {lvl: [lbl for idx, lbl in enumerate(groups[lvl])
                            if bools[lvl][idx]] for lvl in levels}

    # replace with NaN for each level
    print('Drop small groups')
    for lvl in levels:
        amazon_df[lvl].replace(invalid_groups[lvl], np.nan, inplace=True)

    # remove rows with all NaN
    amazon_df.dropna(axis=0, how='all', subset=levels, inplace=True)

    # test data frame by sampling at each level
    print('Generate test dataframe')
    test_dfs = {}
    for lvl in levels:
        test_dfs[lvl] = amazon_df.groupby(lvl)\
                                 .apply(lambda x: x.sample(samples[lvl]))\
                                 .reset_index(level=0, drop=True)

    # merge all test_dfs
    test_df = pd.concat([test_dfs[lvl] for lvl in levels], join='outer', axis=0)

    # obtain train data frame after dropping test rows
    print('Generate train dataframe')
    train_df = amazon_df.drop(list(test_df.index.values))

    # save data frames to file
    print('Save dataframes to file')
    with open(join(data_path, 'train.csv'), 'w', encoding='latin1') as f:
        train_df.to_csv(f)

    with open(join(data_path, 'test.csv'), 'w', encoding='latin1') as f:
        test_df.to_csv(f)


if __name__ == '__main__':
    main()
