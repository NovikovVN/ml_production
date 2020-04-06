import pandas as pd
import time
from datetime import datetime, timedelta
from zipfile import ZipFile


def time_format(sec):
    return str(timedelta(seconds=sec))


def build_dataset_raw(churned_start_date='2019-01-01',
                      churned_end_date='2019-02-01',
                      inter_list=[(1,7),(8,14)],
                      raw_data_path='train/',
                      dataset_path='dataset/',
                      gdrive_path='',
                      mode='train'):

    start_t = time.time()

    zf = ZipFile(gdrive_path + '{}.zip'.format(raw_data_path[:-1]))
    kwargs = {'sep': ';', 'na_values': ['\\N', 'None'], 'encoding': 'utf-8'}

    sample = pd.read_csv(zf.open('{}sample.csv'.format(raw_data_path)), **kwargs)
    profiles = pd.read_csv(zf.open('{}profiles.csv'.format(raw_data_path)), **kwargs)
    payments = pd.read_csv(zf.open('{}payments.csv'.format(raw_data_path)), **kwargs)
    reports = pd.read_csv(zf.open('{}reports.csv'.format(raw_data_path)), **kwargs)
    abusers = pd.read_csv(zf.open('{}abusers.csv'.format(raw_data_path)), **kwargs)
    logins = pd.read_csv(zf.open('{}logins.csv'.format(raw_data_path)), **kwargs)
    pings = pd.read_csv(zf.open('{}pings.csv'.format(raw_data_path)), **kwargs)
    sessions = pd.read_csv(zf.open('{}sessions.csv'.format(raw_data_path)), **kwargs)
    shop = pd.read_csv(zf.open('{}shop.csv'.format(raw_data_path)), **kwargs)

    print('Run time (reading csv files): {}'.format(time_format(time.time()-start_t)))
#-----------------------------------------------------------------------------------------------------
    print('NO dealing with outliers, missing values and categorical features...')
#-----------------------------------------------------------------------------------------------------
    # На основании дня отвала (last_login_dt) строим признаки, которые описывают активность игрока перед уходом

    print('Creating dataset...')
    # Создадим пустой датасет - в зависимости от режима построения датасета - train или test
    if mode == 'train':
        dataset = sample.copy()[['user_id', 'is_churned', 'level', 'donate_total']]
    elif mode == 'test':
        dataset = sample.copy()[['user_id', 'level', 'donate_total']]

    # Пройдемся по всем источникам, содержащим "динамичекие" данные
    for df in [payments, reports, abusers, logins, pings, sessions, shop]:

        # Получим 'day_num_before_churn' для каждого из значений в источнике для определения недели
        data = pd.merge(sample[['user_id', 'login_last_dt']], df, on='user_id')
        data['day_num_before_churn'] = 1 + (data['login_last_dt'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d')) -
                                data['log_dt'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))).apply(lambda x: x.days)
        df_features = data[['user_id']].drop_duplicates().reset_index(drop=True)

        # Для каждого признака создадим признаки для каждого из времененно интервала (в нашем примере 4 интервала по 7 дней)
        features = list(set(data.columns) - set(['user_id', 'login_last_dt', 'log_dt', 'day_num_before_churn']))
        print('Processing with features:', features)
        for feature in features:
            for i, inter in enumerate(inter_list):
                inter_df = data.loc[data['day_num_before_churn'].between(inter[0], inter[1], inclusive=True)].\
                                groupby('user_id')[feature].mean().reset_index().\
                                rename(index=str, columns={feature: feature+'_{}'.format(i+1)})
                df_features = pd.merge(df_features, inter_df, how='left', on='user_id')

        # Добавляем построенные признаки в датасет
        dataset = pd.merge(dataset, df_features, how='left', on='user_id')

        print('Run time (calculating features): {}'.format(time_format(time.time()-start_t)))

    # Добавляем "статические" признаки
    dataset = pd.merge(dataset, profiles, on='user_id')
#---------------------------------------------------------------------------------------------------------------------------
    dataset.to_csv('{}dataset_raw_{}.csv'.format(gdrive_path + dataset_path, mode), sep=';', index=False)
    print('Dataset is successfully built and saved to {}, run time "build_dataset_raw": {}'.\
          format(dataset_path, time_format(time.time()-start_t)))


if __name__ == '__main__':
    pass
