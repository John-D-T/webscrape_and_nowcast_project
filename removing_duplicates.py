
import pandas as pd
from functools import wraps
import time

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


@wraps
def worker():
    list_of_cinemas = '2023-03-07_list_of_cinemas'
    # london_cinemas_csv = pd.read_csv(list_of_cinemas)

    # TODO - add de-duplication

    # TODO - group all of London into one? Or categorize by postcode


if __name__ == "__main__":
    worker()