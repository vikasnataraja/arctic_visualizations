import os
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

def format_time(total_seconds):
    """
    Convert seconds to hours, minutes, seconds, and milliseconds.

    Parameters:
    - total_seconds: The total number of seconds to convert.

    Returns:
    - A tuple containing hours, minutes, seconds, and milliseconds.
    """
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = (total_seconds - int(total_seconds)) * 1000

    return (int(hours), int(minutes), int(seconds), int(milliseconds))
