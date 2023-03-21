import time

def timer(func):
    def wrapper(*args, **kwargs):
        before = time.time()
        func(*args, **kwargs)
        print(f"Function {func.__name__} took: {time.time() - before:.2f} seconds")
    return wrapper
