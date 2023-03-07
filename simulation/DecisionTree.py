from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
import joblib
import random
import pickle

# svcs = ['adservice','cartservice','checkoutservice','currencyservice','emailservice','frontend','paymentservice','productcatalogservice','recommendationservice','shippingservice']
svcs = ['ts-admin-basic-info-service', 'ts-admin-order-service', 'ts-admin-route-service', 'ts-admin-travel-service', 'ts-admin-user-service', 'ts-assurance-service', 'ts-auth-service', 'ts-avatar-service', 'ts-basic-service', 'ts-cancel-service', 'ts-config-service', 'ts-consign-price-service', 'ts-consign-service', 'ts-contacts-service', 'ts-delivery-service', 'ts-execute-service', 'ts-food-map-service', 'ts-food-service', 'ts-inside-payment-service', 'ts-news-service', 'ts-notification-service', 'ts-order-other-service', 'ts-order-service', 'ts-payment-service', 'ts-preserve-other-service', 'ts-preserve-service', 'ts-price-service', 'ts-rebook-service', 'ts-route-plan-service', 'ts-route-service', 'ts-seat-service', 'ts-security-service', 'ts-station-service', 'ts-ticket-office-service', 'ts-ticketinfo-service', 'ts-train-service', 'ts-travel-plan-service', 'ts-travel-service', 'ts-travel2-service', 'ts-ui-dashboard', 'ts-user-service', 'ts-verification-code-service', 'ts-voucher-service']

def data_loader(path):
    df = pd.read_csv(path)
    cols = [col for col in df.columns if col.endswith('&qps') or col.endswith('&count') or col == 'slo_reward']
    df = df[cols].fillna(0)
    # build dataset
    datas = []
    for _, row in df.iterrows():
        x=[]
        for i in range(len(svcs)):
            svc = svcs[i]
            try:
                x.extend([i, row[svc+'&qps'], row[svc+'&count']])
            except:
                x.extend([i, 0, row[svc+'&count']])
        x.append(row['slo_reward'])
        datas.append(x)
    random.shuffle(datas)
    datas = np.array(datas)
    datas_x, datas_y = datas[:, 0:-1], datas[:,-1]
    return datas_x, datas_y


def setup_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    
if __name__ == '__main__':
    setup_seed(20)
    model = DecisionTreeClassifier()
    datas_x, datas_y = data_loader('train-ticket.csv')
    # datas_x, datas_y = data_loader('../train_data/boutique/real_trace_5s_2.0.csv')
    X_train, X_test, y_train, y_test = train_test_split(datas_x, datas_y)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Test set score:{:.2f}".format(model.score(X_test, y_test)))
    print('acc', accuracy_score(y_test, y_pred))
    print('recall', recall_score(y_test, y_pred))
    print('auc', roc_auc_score(y_test, y_pred))

    fpr, tpr, thresholds = roc_curve(y_test,y_pred,pos_label=None,sample_weight=None,drop_intermediate=True)
    res = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
    pickle.dump(res, open('train_ticket/dt.pkl', 'wb'))
    # joblib.dump(model, 'SVM.model')


    