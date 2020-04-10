from imblearn.over_sampling import SMOTE

from utils.helper import *
from utils.settings import GDRIVE_PATH, INTER_LIST
from utils.WOE_IV import data_vars


def prepare_dataset(dataset,
                    dataset_type='train',
                    dataset_path='dataset/',
                    inter_list=[(1,7),(8,14)],
                    gdrive_path=''):
    print('\n{} DATASET\n'.format(dataset_type.upper()))
    start_t = time.time()
    print('Dealing with missing values, outliers, categorical features...')

    # Профили
    dataset['age'] = dataset['age'].fillna(dataset['age'].median())
    dataset['gender'] = dataset['gender'].fillna(dataset['gender'].mode()[0])
    dataset.loc[~dataset['gender'].isin(['M', 'F']), 'gender'] = dataset['gender'].mode()[0]
    dataset['gender'] = dataset['gender'].map({'M': 1., 'F':0.})
    dataset.loc[(dataset['age'] > 80) | (dataset['age'] < 7), 'age'] = round(dataset['age'].median())
    dataset.loc[dataset['days_between_fl_df'] < -1, 'days_between_fl_df'] = -1
    # Пинги
    for period in range(1, len(inter_list)+1):
        col = 'avg_min_ping_{}'.format(period)
        dataset.loc[(dataset[col] < 0) |
                    (dataset[col].isnull()), col] = dataset.loc[dataset[col] >= 0][col].median()
    # Сессии и прочее
    dataset.fillna(0, inplace=True)
    dataset.to_csv('{}dataset_{}.csv'.format(gdrive_path + dataset_path, dataset_type), sep=';', index=False)

    print('Dataset is successfully prepared and saved to {}, run time (dealing with bad values): {}'.\
          format(dataset_path, time_format(time.time()-start_t)))


class WOETransformer():
    def __init__(self, ratio=0.5, random_state=None, smote=True):
        self.ratio = ratio
        self.random_state = random_state
        self.smote = smote
        self.iv = None
        self.woe_info = {}

    def fit(self, X, y=None):
        start_t = time.time()
        iv_df, self.iv = data_vars(X, y)
        print('Run time (calculating IV): {}'.format(time_format(time.time()-start_t)))

        for var in X.columns:
            small_df = iv_df.loc[iv_df['VAR_NAME'] == var]
            if type(small_df.loc[~small_df['MIN_VALUE'].isnull()]['MIN_VALUE'].values[0]) == str:
                small_df.loc[small_df['MIN_VALUE'].isnull(), 'MIN_VALUE'] = 'NaN'
                small_df.loc[small_df['MAX_VALUE'].isnull(), 'MAX_VALUE'] = 'NaN'
            else:
                small_df.loc[small_df['MIN_VALUE'].isnull(), 'MIN_VALUE'] = 0.
                small_df.loc[small_df['MAX_VALUE'].isnull(), 'MAX_VALUE'] = 0.
            transform_dict = dict(zip(small_df['MAX_VALUE'], small_df['WOE']))
            replace_cmd = ''
            replace_cmd1 = ''
            for i in sorted(transform_dict.items()):
                replace_cmd += str(i[1]) + ' if x <= ' + str(i[0]) + ' else '
                replace_cmd1 += str(i[1]) + ' if x == "' + str(i[0]) + '" else '
            replace_cmd += '0'
            replace_cmd1 += '0'

            self.woe_info[var] = (replace_cmd, replace_cmd1)

        print('Run time (gathering WOE by vars): {}'.format(time_format(time.time()-start_t)))

        return self


    def transform(self, X, y=None):
        start_t = time.time()

        X_WOE = X.copy()
        for var in X_WOE.columns:
            replace_cmd, replace_cmd1 = self.woe_info[var]

            if replace_cmd != '0':
                try:
                   X_WOE[var] = X_WOE[var].apply(lambda x: eval(replace_cmd))
                except:
                   X_WOE[var] = X_WOE[var].apply(lambda x: eval(replace_cmd1))

        print('Run time (encoding vars with WOE): {}'.format(time_format(time.time()-start_t)))

        return X_WOE


    def fit_transform(self, X, y):
        self.fit(X, y)
        X_WOE = self.transform(X, y)

        if self.smote:
            start_t = time.time()
            X_WOE_balanced, y_balanced = SMOTE(random_state=self.random_state,
                                               ratio=self.ratio).fit_sample(X_WOE, y)
            print('Run time (oversampling with SMOTE): {}'.format(time_format(time.time()-start_t)))
            return X_WOE_balanced, y_balanced

        return X_WOE, y


if __name__ == '__main__':

    train = pd.read_csv(GDRIVE_PATH + 'dataset/dataset_raw_train.csv', sep=';')
    test = pd.read_csv(GDRIVE_PATH + 'dataset/dataset_raw_test.csv', sep=';')

    prepare_dataset(train, dataset_type='train', inter_list=INTER_LIST, gdrive_path=GDRIVE_PATH)
    prepare_dataset(test, dataset_type='test', inter_list=INTER_LIST, gdrive_path=GDRIVE_PATH)

