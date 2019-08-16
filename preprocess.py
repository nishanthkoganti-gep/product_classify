# import modules
import random
import argparse
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from os.path import join, realpath, dirname

# variables for performing text preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
spl_chars = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~\'\t\n'


def proc_lbl(lbl_list):
    # obtain # of levels in labels
    if len(lbl_list) > 0:
        num_levels = np.array([len(lbl.split(', ')) for lbl in lbl_list])
        return lbl_list[np.argmax(num_levels)]
    else:
        return None


def proc_str(text, desc=True):
    # remove splchars (need to make faster)
    text = ''.join([ch for ch in text if ch not in spl_chars])

    # apply word tokenizer
    tokens = word_tokenize(text)

    # convert to lower case
    tokens = [token.lower() for token in tokens]

    # apply lemmatizer
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # remove stop words
    tokens = [token for token in tokens if token not in stop_words]

    # apply additional processing for description
    if desc:
        if len(tokens) > 20:
            tokens = random.sample(tokens, k=20)
        random.shuffle(tokens)
    else:
        if len(tokens) > 20:
            tokens = tokens[:20]

    # generate comment_text and comment_pos
    proc_text = ' '.join(tokens)

    return proc_text


def main():
    # setup argument parser
    parser = argparse.ArgumentParser(description='preprocess amazon-670k')
    parser.add_argument('--preload', default=False, action='store_true',
                        help='flag to preload csv files')
    args = parser.parse_args()

    # setup directories
    root_dir = dirname(realpath(__file__))
    data_path = join(root_dir, 'data')

    if args.preload:
        # load the csv files
        with open(join(data_path, 'labels.csv'),
                  'r', encoding='latin1') as f:
            labels_df = pd.read_csv(f)

        with open(join(data_path, 'titles.csv'),
                  'r', encoding='latin1') as f:
            titles_df = pd.read_csv(f)

        with open(join(data_path, 'descriptions.csv'),
                  'r', encoding='latin1') as f:
            descriptions_df = pd.read_csv(f)

    else:
        # load the labels text
        print('Reading Labels')
        with open(join(data_path, 'labels.txt'),
                  'r', encoding='latin1') as f:
            label_text = f.read()
        label_lines = label_text.split('\n')

        # loop over the label text
        print('Extracting Labels')
        pid = None
        labels = {}
        for line in label_lines:
            if line[:2] != '  ':
                pid = line
                labels[pid] = []
            else:
                labels[pid].append(line[2:])

        # assign single labels with most levels
        print('Processing Labels')
        proc_labels = {}
        for pid in labels:
            lbl_list = labels[pid]
            lbl = proc_lbl(lbl_list)
            if lbl:
                proc_labels[pid] = lbl

        # saving labels to file
        print('Saving Labels')
        labels_df = pd.DataFrame(list(proc_labels.items()),
                                 columns=['pid', 'label'])

        # split to different levels
        labels_df['level1'] = labels_df['label'].apply(lambda x:
                                                       x.split(', ')[0])
        labels_df['level2'] = labels_df['label'].apply(lambda x:
                                                       ', '.join(x.split(', ')[:2])
                                                       if len(x.split(', ')) > 1 else '')
        labels_df['level3'] = labels_df['label'].apply(lambda x:
                                                       ', '.join(x.split(', ')[:3])
                                                       if len(x.split(', ')) > 2 else '')
        labels_df['level4'] = labels_df['label'].apply(lambda x:
                                                       ', '.join(x.split(', ')[:4])
                                                       if len(x.split(', ')) > 3 else '')
        labels_df = labels_df.drop(columns=['label'])

        with open(join(data_path, 'labels.csv'), 'w') as f:
            labels_df.to_csv(f, index=False)

        # load the titles text
        print('Reading Titles')
        with open(join(data_path, 'titles.txt'),
                  'r', encoding='latin1') as f:
            title_text = f.read()
        title_samples = title_text.split('\n')

        # loop over the titles
        print('Extracting Titles')
        counter = 0
        titles = {}
        for sample in title_samples:
            counter += 1
            line = sample.split(' ')
            if len(line) > 1:
                # process the title text
                title_txt = ' '.join(line[1:])
                title_txt = proc_str(title_txt, desc=False)

                # append to dictionaries
                titles[line[0]] = title_txt
            else:
                print('Err: ', sample)
                continue

            if counter % 50000 == 0:
                print(f'Count: {counter}, Text: {title_txt}')

        # saving titles to pickle file
        print('Saving Titles')
        titles_df = pd.DataFrame(list(titles.items()),
                                 columns=['pid', 'title'])
        with open(join(data_path, 'titles.csv'), 'w',
                  encoding='latin1') as f:
            titles_df.to_csv(f, index=False)

        # load the description text
        print('Reading Descriptions')
        with open(join(data_path, 'descriptions.txt'),
                  'r', encoding='latin1') as f:
            descriptions_text = f.read()
        description_lines = descriptions_text.split('\n')

        # loop over the description lines
        print('Extracting Descriptions')
        pid = None
        counter = 0
        descriptions = {}
        for line in description_lines:
            if line[:17] == 'product/productId':
                line = line.split(' ')
                if len(line) > 1:
                    pid = line[1]
                else:
                    continue
            elif line[:19] == 'product/description':
                counter += 1
                line = line.split(' ')
                if len(line) > 1:
                    # process the description text
                    desc_txt = ' '.join(line[1:])
                    desc_txt = proc_str(desc_txt)

                    # append to dictionaries
                    descriptions[pid] = desc_txt
                else:
                    continue

                if counter % 10000 == 0:
                    print(f'Count: {counter}, Text: {desc_txt}')
            else:
                continue

        # saving descriptions to pickle file
        print('Saving Descriptions')
        descriptions_df = pd.DataFrame(list(descriptions.items()),
                                       columns=['pid', 'description'])
        with open(join(data_path, 'descriptions.csv'), 'w',
                  encoding='latin1') as f:
            descriptions_df.to_csv(f, index=False)

    # set product id as index to frames
    labels_df = labels_df.set_index('pid')
    titles_df = titles_df.set_index('pid')
    descriptions_df = descriptions_df.set_index('pid')

    # merge all the dataframes to form a master data frame
    amazon_df = pd.concat([labels_df, titles_df, descriptions_df],
                          axis=1, sort=True)
    amazon_df = amazon_df[amazon_df['level1'].notna()]

    # save master data frame to file
    with open(join(data_path, 'amazon.csv'), 'w', encoding='latin1') as f:
        amazon_df.to_csv(f)


if __name__ == '__main__':
    main()
