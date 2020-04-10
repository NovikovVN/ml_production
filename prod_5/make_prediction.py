import pickle

from prepare_dataset import WOETransformer
from utils.helper import *
from utils.settings import GDRIVE_PATH


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

    answers = pd.DataFrame({'is_churned': predictions.astype('uint8')})

    answers.to_csv(gdrive_path + saving_path + 'predictions.csv', sep=';')
    print('Predictions are successfully saved to {}{}predictions.csv'.\
          format(gdrive_path, saving_path))


if __name__ == '__main__':

    dataset_raw_test = pd.read_csv(GDRIVE_PATH + 'dataset/dataset_raw_test.csv', sep=';')
    X_test = dataset_raw_test.drop(['user_id'], axis=1)

    make_prediction(X_test, saving_path='dataset/', gdrive_path=GDRIVE_PATH)

    predictions = pd.read_csv(GDRIVE_PATH + 'dataset/predictions.csv', index_col=[0], sep=';')
    answers = pd.concat([dataset_raw_test['user_id'], predictions], axis=1)
    answers.to_csv(GDRIVE_PATH + 'dataset/answers.csv', sep=';')

