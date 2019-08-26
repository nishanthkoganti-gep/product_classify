# import modules
import pandas as pd
from os.path import join, realpath, dirname
from IPython.terminal.debugger import set_trace as keyboard


def main():
    # setup data directory
    root_dir = dirname(realpath(__file__))
    data_path = join(root_dir, '..', 'data', 'amazon')

    # load the amazon csv
    with open(join(data_path, 'amazon.csv'),
              'r', encoding='latin1') as f:
        amazon_df = pd.read_csv(f)
    amazon_df.rename(columns={'Unnamed: 0': 'pid'},
                     inplace=True)

    # include additional columns in amazon csv
    amazon_df = amazon_df.set_index('pid')
    amazon_df['related_title'] = ''
    amazon_df['related_description'] = ''

    # load the related text file
    with open(join(data_path, 'related.txt'),
              'r', encoding='latin1') as f:
        related_text = f.read()
        related_lines = related_text.split('\n')

    # obtain product ids
    valid_df = amazon_df[amazon_df['title'].notna() & amazon_df['description'].notna()]
    pids = list(valid_df.index.values)
    titles = valid_df['title'].to_dict()
    descriptions = valid_df['description'].to_dict()

    related_pids = {}
    related_titles = {}
    related_descriptions = {}

    # loop over the rows and assign values
    counter = 0
    for line in related_lines:
        # process string input
        item = line.split(' also purchased ')

        # obtain pid and related items
        pid = item[0]
        related = [rel_pid for rel_pid in item[1].split(' ')
                   if rel_pid != 'rights_details']

        # loop over related items
        for rel_pid in related:
            if rel_pid in pids:
                related_pids[pid] = rel_pid
                related_titles[pid] = titles[rel_pid]
                related_descriptions[pid] = descriptions[rel_pid]
                break

        # counter analysis
        counter += 1
        if counter % 100 == 0:
            print(f'Counter: {counter}')

    keyboard()

    # write to amazon csv
    with open(join(data_path, 'amazon_full.csv'),
              'w', encoding='latin1') as f:
        amazon_df.to_csv(f)


if __name__ == '__main__':
    main()
