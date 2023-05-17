import pandas as pd
import os
import glob

def get_average_imdb_ratings_all_movies():
    # 368,000 movies pre data cleaning
    pd.set_option('display.max_rows', 500)

    imdb_df = pd.read_csv(os.path.join(os.getcwd(), 'genre', 'all_movies.csv'))
    imdb_df = imdb_df[imdb_df['votes'].notna()]
    imdb_df['votes'] = imdb_df['votes'].replace({'K': '000', 'M': '000000'}, regex=True).map(pd.eval).astype(int)
    imdb_df['user_rating_adjusted'] = pd.to_numeric(imdb_df['votes'], errors='coerce').fillna(0)
    imdb_df = imdb_df[imdb_df['user_rating_adjusted'] > 5000]

    imdb_df['rating_adjusted'] = pd.to_numeric(imdb_df['rating'], errors='coerce').fillna(0)
    pd.to_numeric(imdb_df['rating_adjusted'])

    imdb_df['year_adjusted'] = pd.to_numeric(imdb_df['year'], errors='coerce').fillna(0)
    pd.to_numeric(imdb_df['year_adjusted'])

    'gross(in $)'


if __name__ == '__main__':
    get_average_imdb_ratings_all_movies()