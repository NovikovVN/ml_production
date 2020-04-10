import pandas as pd
import time
from datetime import datetime, timedelta


def time_format(sec):
    return str(timedelta(seconds=sec))
