import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, auc, \
                            log_loss, roc_auc_score, average_precision_score, confusion_matrix, classification_report

from prepare_dataset import WOETransformer
from utils.helper import *
from utils.settings import GDRIVE_PATH


def evaluation(y_true, y_pred, y_prob):
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    ll = log_loss(y_true=y_true, y_pred=y_prob)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_prob)
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1: {}'.format(f1))
    print('Log Loss: {}'.format(ll))
    print('ROC AUC: {}'.format(roc_auc))
    return precision, recall, f1, ll, roc_auc


def show_proba_calibration_plots(y_true, y_prob):
    preds_with_trues = np.array(list(zip(y_prob, y_true)))

    thresholds = []
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in np.linspace(0.1, 0.9, 9):
        thresholds.append(threshold)
        precisions.append(precision_score(y_true, list(map(int, y_prob > threshold))))
        recalls.append(recall_score(y_true, list(map(int, y_prob > threshold))))
        f1_scores.append(f1_score(y_true, list(map(int, y_prob > threshold))))

    scores_table = pd.DataFrame({'f1':f1_scores,
                                 'precision':precisions,
                                 'recall':recalls,
                                 'probability':thresholds}).sort_values('f1', ascending=False).round(3)

    figure = plt.figure(figsize = (15, 5))

    plt1 = figure.add_subplot(121)
    plt1.plot(thresholds, precisions, label='Precision', linewidth=4)
    plt1.plot(thresholds, recalls, label='Recall', linewidth=4)
    plt1.plot(thresholds, f1_scores, label='F1', linewidth=4)
    plt1.set_ylabel('Scores')
    plt1.set_xlabel('Probability threshold')
    plt1.set_title('Probabilities threshold calibration')
    plt1.legend(bbox_to_anchor=(0.25, 0.25))
    plt1.table(cellText = scores_table.values,
               colLabels = scores_table.columns,
               colLoc = 'center', cellLoc = 'center', loc = 'bottom', bbox = [0, -1.3, 1, 1])

    plt2 = figure.add_subplot(122)
    plt2.hist(preds_with_trues[preds_with_trues[:, 1] == 0][:, 0],
              label='0', color='royalblue', alpha=0.5, density=True)
    plt2.hist(preds_with_trues[preds_with_trues[:, 1] == 1][:, 0],
              label='1', color='orange', alpha=0.5, density=True)
    plt2.set_ylabel('Density')
    plt2.set_xlabel('Probabilities')
    plt2.set_title('Probability histogram')
    plt2.legend(bbox_to_anchor=(1, 1))

    plt.show()


def train_model(X, y, random_state=None, saving_path='model/', gdrive_path=''):
    start_t = time.time()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          test_size=0.1,
                                                          shuffle=True,
                                                          stratify=y,
                                                          random_state=random_state)
    print('Splitting data on train and valid: {}'.format(time_format(time.time()-start_t)))

    print('Train WOE transformation...')
    woe_transformer = WOETransformer(random_state=random_state, ratio=0.5)
    X_train_woe, y_train_woe = woe_transformer.fit_transform(X_train, y_train)

    print('Fitting LogisticRegressionCV...')
    logit = LogisticRegressionCV(cv=5, scoring='f1',
                                 solver='liblinear', random_state=random_state, n_jobs=-1)
    logit.fit(X_train_woe, y_train_woe)
    print('Run time (fitting and CV model): {}'.format(time_format(time.time()-start_t)))

    print('Valid WOE transformation...')
    X_valid_woe = woe_transformer.transform(X_valid)

    print('\nVALID METRICS\n')
    y_valid_prob = logit.predict_proba(X_valid_woe)[:, 1]
    y_valid_pred = logit.predict(X_valid_woe)
    precision_test, recall_test, f1_test, log_loss_test, roc_auc_test = \
        evaluation(y_valid, y_valid_pred, y_valid_prob)

    threshold = 0.8
    print('\nThreshold =', threshold)

    print('\nClassification report with threshold\n'.upper())
    print(classification_report(y_valid, y_valid_prob > threshold))

    with open(gdrive_path + saving_path + 'logit.pickle', 'wb') as f:
        pickle.dump(logit, f)
    with open(gdrive_path + saving_path + 'woe_transformer.pickle', 'wb') as f:
        pickle.dump(woe_transformer, f)
    with open(gdrive_path + saving_path + 'threshold.pickle', 'wb') as f:
        pickle.dump(threshold, f)
    print('Run time (saving logit, woe_transformer and threshold): {}'.format(time_format(time.time()-start_t)))


if __name__ == '__main__':

    dataset_raw_train = pd.read_csv(GDRIVE_PATH + 'dataset/dataset_raw_train.csv', sep=';')
    X_raw = dataset_raw_train.drop(['user_id', 'is_churned'], axis=1)
    y_raw = dataset_raw_train['is_churned']

    train_model(X_raw, y_raw, random_state=42, gdrive_path=GDRIVE_PATH)
