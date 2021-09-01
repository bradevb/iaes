import concurrent.futures
from functools import wraps

_DEFAULT_POOL = concurrent.futures.ThreadPoolExecutor()


def threadpool(f, executor=None):
    @wraps(f)
    def wrap(*args, **kwargs):
        return (executor or _DEFAULT_POOL).submit(f, *args, **kwargs)

    return wrap
