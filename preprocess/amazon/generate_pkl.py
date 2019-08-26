# import modules
import bcolz
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from os.path import dirname, realpath, join
from IPython.terminal.debugger import set_trace as keyboard


# function for tokenizing
def corpus_indexify(corpus_dict, word2idx):
    # initialize corpus array
    token_found = 0
    token_count = 0
    max_length = 20
    n_items = len(corpus_dict)
    corpus_arr = word2idx['PAD'] * np.ones((n_items, max_length))

    # loop over the descriptions
    for idx, key in enumerate(corpus_dict.keys()):
        tokens = corpus_dict[key].split(' ')
        for tidx, token in enumerate(tokens):
            try:
                corpus_arr[idx, tidx] = word2idx[token]
                token_found += 1
            except KeyError:
                corpus_arr[idx, tidx] = word2idx['UNK']
            token_count += 1

    # compute coverage
    coverage = token_found / token_count
    return corpus_arr, coverage


def main():
    parser = argparse.ArgumentParser(description='generate datasets')
    parser.add_argument('--embed_type', type=str, default='glove',
                        help='embedding type')
    parser.add_argument('--embed_dim', type=int, default=300,
                        help='embedding dimension')
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
    data_path = join(root_dir, '..', 'data', 'amazon')
    embed_path = join(root_dir, '..', 'embeddings')

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

    # load embeddings
    print('Load word embeddings')
    embed_dim = args.embed_dim
    embed_type = args.embed_type

    # load word embeddings
    vectors = bcolz.open(f'{embed_path}/{embed_type}.{embed_dim}d.dat')[:]
    words = pickle.load(open(f'{embed_path}/{embed_type}.{embed_dim}d_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{embed_path}/{embed_type}.{embed_dim}d_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}

    # parse through dictionaries
    print('Normalize with training vocabulary')
    train_titles_str = ' '.join(list(train_df['title'].values))
    train_descriptions_str = ' '.join(list(train_df['description'].values))

    # obtain training and test vocabularies
    train_vocab = set(train_titles_str.split(' ') + train_descriptions_str.split(' '))

    # check coverage of vocabularies
    words_found = 0
    train2idx = {}
    weights_matrix = []

    for i, word in enumerate(train_vocab):
        try:
            weights_matrix.append(glove[word])
            train2idx[word] = words_found
            words_found += 1
        except KeyError:
            continue

    # assign unknown and pad tokens
    train2idx['UNK'] = words_found
    train2idx['PAD'] = words_found + 1
    weights_matrix.append(np.random.normal(scale=0.6, size=(embed_dim)))
    weights_matrix.append(np.random.normal(scale=0.6, size=(embed_dim)))

    # convert to weight matrix
    weights_matrix = np.array(weights_matrix)

    # save embedding matrix and weights to file
    with open(join(embed_path, f'amazon.{embed_type}.{embed_dim}.csv'), 'w') as f:
        np.savetxt(f, weights_matrix, delimiter=',')

    with open(join(embed_path, f'amazon.{embed_type}.{embed_dim}_idx.pkl'), 'wb') as f:
        pickle.dump(train2idx, f)

    # obtain titles and descriptions as dictionaries
    print('Generate dataset pickle files')
    train_titles = train_df['title'].to_dict()
    train_descriptions = train_df['description'].to_dict()

    test_titles = test_df['title'].to_dict()
    test_descriptions = test_df['description'].to_dict()

    # obtain indices
    train_title_arr, train_title_cov = corpus_indexify(train_titles, train2idx)
    train_description_arr, train_description_cov = corpus_indexify(train_descriptions, train2idx)

    test_title_arr, test_title_cov = corpus_indexify(test_titles, train2idx)
    test_description_arr, test_description_cov = corpus_indexify(test_descriptions, train2idx)

    # check training data coverage
    print(f'Train title coverage: {train_title_cov}')
    print(f'Train description coverage: {train_description_cov}')
    print(f'Test title coverage: {test_title_cov}')
    print(f'Test description coverage: {test_description_cov}')

    test_data = {'title': test_title_arr, 'description': test_description_arr}
    train_data = {'title': train_title_arr, 'description': train_description_arr}

    # generate label arrays
    print('Generate label arrays')
    levels = ['level1', 'level2', 'level3', 'level4']

    for lvl in levels:
        # generate lists of labels
        lbls = list(set(list(train_df[lvl].values)))
        lbls.remove(np.nan)

        # generate dict mapping
        level2idx = {lbl: idx for idx, lbl in enumerate(lbls)}
        level2idx[np.nan] = -1

        test_data[lvl] = test_df[lvl].apply(lambda x, level2idx=level2idx: level2idx[x])
        train_data[lvl] = train_df[lvl].apply(lambda x, level2idx=level2idx: level2idx[x])

    # save all arrays to pickle files
    print('Save to pickle files')

    with open(join(data_path, 'train.pkl'), 'wb') as f:
        pickle.dump(train_data, f)

    with open(join(data_path, 'test.pkl'), 'wb') as f:
        pickle.dump(test_data, f)


if __name__ == '__main__':
    main()
