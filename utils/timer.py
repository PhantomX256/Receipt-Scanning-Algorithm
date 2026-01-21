import time
import functools

from utils.logger import Logger


def timer(function):
    """
    Logs the runtime of the decorated function
    """
    @functools.wraps(function)
    def wrapper_timer(*args, **kwargs):
        logger = Logger()

        # Record the start time
        start_time = time.perf_counter()

        # Execute the function
        value = function(*args, **kwargs)

        # Record end time and time taken
        end_time = time.perf_counter()
        run_time = end_time - start_time

        logger.info(f"Finished {function.__name__}() in {run_time:.4f} secs")
        return value

    return wrapper_timer