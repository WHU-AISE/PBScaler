import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import seaborn as sns
import joblib


# svcs = ['adservice','cartservice','checkoutservice','currencyservice','emailservice','frontend','paymentservice','productcatalogservice','recommendationservice','shippingservice']

def data_loader(path):
    df = pd.read_csv(path)
    cols = [col for col in df.columns if col.endswith('&qps') or col.endswith('&count') or col == 'reward']
    df = df[cols].dropna(axis=0, how='any')
    svcs = [col.replace('&qps', '') for col in df.columns if col.endswith('&qps')]
    svcs.sort()
    # build dataset
    datas_x = []
    datas_y = []
    for _, row in df.iterrows():
        x = []
        for i in range(len(svcs)):
            svc = svcs[i]
            x.extend([i, row[svc + '&qps'], row[svc + '&count']])
        datas_y.append(row['reward'])
        datas_x.append(x)
    datas_x = np.array(datas_x)
    datas_y = np.array(datas_y)
    return datas_x, datas_y


if __name__ == '__main__':
    params = {
        "n_estimators": [20, 50, 80, 100],
        "max_depth": [2, 4, None]
    }
    model = LinearRegression()
    datas_x, datas_y = data_loader('../train_data/train_ticket/train-ticket.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        datas_x, datas_y, test_size=0.2, random_state=10)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # print("Test set score:{:.2f}".format(model.score(X_test, y_test)))
    # print('acc', accuracy_score(y_test, y_pred))
    # print('auc', roc_auc_score(y_test, y_pred))
    print('MAE', mean_absolute_error(y_test, y_pred))
    print('MSE', mean_squared_error(y_test, y_pred))
    print('r2', r2_score(y_test, y_pred))
    plot_df = pd.DataFrame({'true': y_test[0:50], 'pred': y_pred[0:50]})
    # fig = sns.lineplot(data=plot_df)
    # fig.get_figure().savefig('RandomForest.png')
    sns.set(color_codes=True)
    p = sns.regplot(x="true", y="pred", data=plot_df)
    plt.legend(labels=["predict", "true"])
    p.get_figure().savefig('Linear_regression.png')
    # joblib.dump(model, 'RandomForest.model')

# %%
