import numpy as np
import pandas as pd
import pickle
import time
from datetime import datetime, timedelta

from prepare_dataset import *


def time_format(sec):
    return str(timedelta(seconds=sec))


def make_prediction(X, loading_path='model/', saving_path='dataset/', gdrive_path=''):
    start_t = time.time()

    with open(gdrive_path + loading_path + 'logit.pickle', 'rb') as f:
        logit = pickle.load(f)
    with open(gdrive_path + loading_path + 'woe_transformer.pickle', 'rb') as f:
        woe_transformer = pickle.load(f)
    with open(gdrive_path + loading_path + 'threshold.pickle', 'rb') as f:
        threshold = pickle.load(f)
    print('Run time (loading logit, woe_transformer and threshold): {}'.format(time_format(time.time()-start_t)))

    print('Test WOE transformation...')
    X_test_woe = woe_transformer.transform(X)

    print('Taking predictions...')
    predictions = logit.predict_proba(X_test_woe)[:, 1] > threshold

    answers = pd.DataFrame({'is_churned': predictions.astype(np.uint8)})

    answers.to_csv(gdrive_path + saving_path + 'predictions.csv', sep=';')
    print('Predictions are successfully saved to {}{}predictions.csv'.\
          format(gdrive_path, saving_path))

if __name__ == '__main__':
    pass
