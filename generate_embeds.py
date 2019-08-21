# import modules
import bcolz
import pickle
import argparse
import numpy as np
import pandas as pd
from os.path import realpath, dirname, join


def main():
    # setup argument parser
    parser = argparse.ArgumentParser(description='generate embedding pickle file')
    parser.add_argument('--embed_type', type=str, default='glove', help='embedding type')
    parser.add_argument('--embed_dim', type=int, default=300, help='embedding dimension')
    parser.add_argument('--embed_tokens', type=int, default=6, help='embedding tokens')
    args = parser.parse_args()

    print('Initialize variables')

    # get embedding details
    embed_dim = args.embed_dim
    embed_type = args.embed_type
    embed_tokens = args.embed_tokens

    # setup directory name
    root_dir = dirname(realpath(__file__))
    embed_path = join(root_dir, 'embeddings')

    # setup embedding parameters
    idx = 0
    words = []
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1),
                           rootdir=f'{embed_path}/{embed_type}.{embed_dim}d.dat',
                           mode='w')

    print('Processing word vectors')

    with open(f'{embed_path}/{embed_type}.{embed_tokens}B.{embed_dim}d.txt', 'rb') as f:
        for l in f:
            # decode line by splitting
            line = l.decode().split()

            # obtain word and vector
            word = line[0]
            vect = np.array(line[1:]).astype(np.float)

            # append to variables
            words.append(word)
            word2idx[word] = idx
            vectors.append(vect)

            idx += 1

            # print progress
            if idx % 10000 == 0:
                print(f'Count: {idx}, Word: {word}')

    print('Save word vectors to file')

    # write vectors to file
    vectors = bcolz.carray(vectors[1:].reshape((idx, embed_dim)),
                           rootdir=f'{embed_path}/{embed_type}.{embed_dim}d.dat',
                           mode='w')
    vectors.flush()

    # dump words and word dictionary to file
    pickle.dump(words, open(f'{embed_path}/{embed_type}.{embed_dim}d_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{embed_path}/{embed_type}.{embed_dim}d_idx.pkl', 'wb'))


if __name__ == '__main__':
    main()
