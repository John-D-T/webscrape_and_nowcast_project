"""
Compile all individual scraped twitter .csvs into one csv
"""

import os
import glob
import pandas as pd

def merge_csv_files(input_dir_path, output_file_path):
    os.chdir(input_dir_path)
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    combined_csv.to_csv(output_file_path, index=False, encoding='utf-8-sig')



if __name__ == '__main__':

    input_dir_path = os.path.join(os.getcwd(), 'twitter_output')
    output_file_path = os.path.join(os.getcwd(), 'compiled_tweets_odeon.csv')

    merge_csv_files(input_dir_path, output_file_path)