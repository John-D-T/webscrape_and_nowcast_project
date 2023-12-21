"""
PYTHON 3.11 (64 BIT)
"""

import pandas as pd
import time
import os


def timeit(func):
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


@timeit
def worker(input_csv, output_csv):
    """
    Function to take in an unrefined csv of cinemas, and then drop duplicates and remove irrelevant data, and
    return a refined csv of cinemas
    :param input_csv: path of unrefined csv of cinemas
    :param output_csv: path of refined csv of cinemas
    """
    london_cinemas_df = pd.read_csv(os.path.join(os.getcwd(), input_csv))
    london_cinemas_refined_df = london_cinemas_df.drop_duplicates(subset=['cinema_name', 'cinema_url'])
    london_cinemas_refined_df = london_cinemas_refined_df[(london_cinemas_refined_df.category == 'Cinema') |
                                                           (london_cinemas_refined_df.category == 'Outdoor cinema') |
                                                           (london_cinemas_refined_df.category == 'Imax Cinema') |
                                                           (london_cinemas_refined_df.category == 'Film production cinema') |
                                                           (london_cinemas_refined_df.category == 'Arts Organisation') |
                                                           (london_cinemas_refined_df.category == 'Events Venue')]

    # write to csv file
    london_cinemas_refined_df.to_csv(os.path.join(os.getcwd(), output_csv))


if __name__ == "__main__":
    worker(input_csv='output\\2023-03-08_list_of_cinemas.csv',
           output_csv='output\\2023-03-08_list_of_cinemas_refined.csv')
