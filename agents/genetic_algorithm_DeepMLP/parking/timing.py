from functools import wraps
from time import time
import sys


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print(f'Elapsed time: {(end - start):.3f}s')
        return result

    return wrapper
