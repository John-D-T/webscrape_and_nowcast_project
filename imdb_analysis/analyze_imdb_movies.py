import pandas as pd
import os
import glob

def get_average_imdb_ratings_25k_movies():
    pd.set_option('display.max_rows', 500)

    imdb_df = pd.read_csv(os.path.join(os.getcwd(), '25k_imdb_movie_dataset.csv'))

    imdb_df['User Rating'] = imdb_df['User Rating'].replace({'K': '000', 'M': '000000'}, regex=True).map(pd.eval).astype(int)
    pd.to_numeric(imdb_df['User Rating'])
    imdb_df = imdb_df[imdb_df['user_rating_adjusted'] > 5000]
    imdb_df['user_rating_adjusted'] = pd.to_numeric(imdb_df['User Rating'], errors='coerce').fillna(0)

    imdb_df['rating_adjusted'] = pd.to_numeric(imdb_df['Rating'], errors='coerce').fillna(0)
    pd.to_numeric(imdb_df['rating_adjusted'])

    imdb_df['year_adjusted'] = pd.to_numeric(imdb_df['year'], errors='coerce').fillna(0)
    pd.to_numeric(imdb_df['year_adjusted'])
    imdb_df['year_adjusted'] = imdb_df['year_adjusted'].abs()
    imdb_df = imdb_df[imdb_df['year_adjusted'] > 1999]
    # https://stackoverflow.com/questions/44522741/pandas-mean-typeerror-could-not-convert-to-numeric
    imdb_df_grouped = imdb_df.groupby('year_adjusted')['rating_adjusted'].mean()

    save_df_as_image(df=imdb_df_grouped, file_name='imdb_rating_25k')

def get_average_imdb_ratings_all_movies():
    # 368,000 movies pre data cleaning
    pd.set_option('display.max_rows', 500)

    imdb_df = pd.read_csv(os.path.join(os.getcwd(), 'genre', 'all_movies.csv'))
    imdb_df = imdb_df[imdb_df['votes'].notna()]
    imdb_df['votes'] = imdb_df['votes'].replace({'K': '000', 'M': '000000'}, regex=True).map(pd.eval).astype(int)
    imdb_df['user_rating_adjusted'] = pd.to_numeric(imdb_df['votes'], errors='coerce').fillna(0)
    # imdb_df = imdb_df[imdb_df['user_rating_adjusted'] > 5000]

    imdb_df = imdb_df[imdb_df['gross(in $)'] > 1000000]

    imdb_df['rating_adjusted'] = pd.to_numeric(imdb_df['rating'], errors='coerce').fillna(0)
    pd.to_numeric(imdb_df['rating_adjusted'])

    imdb_df['year_adjusted'] = pd.to_numeric(imdb_df['year'], errors='coerce').fillna(0)
    pd.to_numeric(imdb_df['year_adjusted'])
    imdb_df['year_adjusted'] = imdb_df['year_adjusted'].abs()
    imdb_df = imdb_df[imdb_df['year_adjusted'] > 1999]

    # https://stackoverflow.com/questions/44522741/pandas-mean-typeerror-could-not-convert-to-numeric
    imdb_df_grouped = imdb_df.groupby('year_adjusted')['rating_adjusted'].mean().reset_index()

    save_df_as_image(df=imdb_df_grouped, file_name='imdb_rating_368k_v3')

def save_df_as_image(df, file_name):
    import subprocess

    filename = '%s.tex' % file_name
    pdffile = '%s.pdf' % file_name
    outname = '%s.png' % file_name

    template = r'''\documentclass[preview]{{standalone}}
    \usepackage{{booktabs}}
    \begin{{document}}
    {}
    \end{{document}}
    '''

    with open(filename, 'w+') as f:
        f.write(template.format(df.to_latex()))

    subprocess.call(['pdflatex', filename])
    subprocess.call(['convert', '-density', '300', pdffile, '-quality', '90', outname])

def concatenate_imdb_movies():
    extension = 'csv'
    path = os.path.join(os.getcwd(), 'genre')
    os.chdir(path)
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    # export to csv
    combined_csv.to_csv("all_movies.csv", index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    #get_average_imdb_ratings_25k_movies()
    get_average_imdb_ratings_all_movies()