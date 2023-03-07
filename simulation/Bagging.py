import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import joblib

svcs = ['adservice','cartservice','checkoutservice','currencyservice','emailservice','frontend','paymentservice','productcatalogservice','recommendationservice','shippingservice']

def data_loader(path):
    df = pd.read_csv(path).dropna(axis=0, how='any')
    # build dataset
    datas_x = []
    datas_y = []
    for _, row in df.iterrows():
        x=[]
        for i in range(len(svcs)):
            svc = svcs[i]
            x.extend([i, row[svc+'&qps'], row[svc+'&count']])
        datas_y.append(row['reward'])
        datas_x.append(x)
    datas_x = np.array(datas_x)
    datas_y = np.array(datas_y)
    return datas_x, datas_y
    
if __name__ == '__main__':
    params = {
        "n_estimators": [10, 40, 70, 100],
        "base_estimator": [None]
    }
    model = BaggingRegressor()
    grid_search = GridSearchCV(model, params, cv=5)
    datas_x, datas_y = data_loader('../train_data/real_trace_5s_3.0.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        datas_x, datas_y, test_size=0.2, random_state=10)

    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)

    print("Test set score:{:.2f}".format(grid_search.score(X_test, y_test)))
    print("Best parameters:{}".format(grid_search.best_params_))
    print("Best score on train set:{:.2f}".format(grid_search.best_score_))
    print('MSE', mean_squared_error(y_test, y_pred))
    print('MAE', mean_absolute_error(y_test, y_pred))
    print('R2', r2_score(y_test, y_pred))
    plot_df = pd.DataFrame({'true': y_test[0:50], 'pred': y_pred[0:50]})
    fig = sns.lineplot(data=plot_df)
    fig.get_figure().savefig('bagging.png')