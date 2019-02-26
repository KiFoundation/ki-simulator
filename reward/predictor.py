import os
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from datetime import datetime


def basic_prediction():
    return


def forecasting(df_blocks_):
    df_blocks_['date'] = pd.to_datetime(df_blocks_['date'])
    df_blocks_ = df_blocks_.set_index('date')
    df_blocks_ = df_blocks_['tx'].resample('W').mean()
    print(len(df_blocks_))

    decomposition = sm.tsa.seasonal_decompose(df_blocks_, model='additive')
    fig = decomposition.plot()

    arima(df_blocks_, False)


def parameter_selection(df_blocks_):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    ress = []

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(df_blocks_,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                ress.append([param, param_seasonal, results.aic])
                # print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue

    return np.asarray(sorted(ress, key=lambda x: x[2], reverse=True))[-1]


def arima(df_blocks_, param_select):

    # Select parameters
    a = parameter_selection(df_blocks_)

    pred_ci = pd.DataFrame()
    pred_uc_m = pd.DataFrame()
    mse =[]
    for i in range(1, 5, 1):
        mod = sm.tsa.statespace.SARIMAX(df_blocks_[:'201' + str(i)], order=a[0],
                                        seasonal_order=a[1],
                                        enforce_stationarity=False, enforce_invertibility=False)
        # Fit the selection model
        results = mod.fit()

        # Plot diagnostics
        results.plot_diagnostics(figsize=(16, 8))

        # Get and plot predictions staring a given date
        pred = results.get_prediction(start=pd.to_datetime(df_blocks_['201' + str(i) + '-02':].index.values[0]), dynamic=False)

        # Compute MSE and RMSE error
        y_forecasted = pred.predicted_mean
        y_truth = df_blocks_['201' + str(i) + '-06':]
        #
        mse.append(((y_forecasted - y_truth) ** 2).mean())
        # print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
        #
        pred_uc = results.get_forecast(steps=1)
        pred_ci = pd.concat([pred_ci, pred_uc.conf_int()])
        pred_uc_m = pd.concat([pred_uc_m, pred_uc.predicted_mean])

    print(mse)
    pred_uc_m.columns = ['forecast']
    print(pred_ci)
    print(pred_uc_m)
    df_ = pd.DataFrame({'date': df_blocks_['2011':'2018'].index.values,
                        'data': [df_blocks_['2011':'2018'][i] for i in df_blocks_['2011':'2018'].index.values]})

    ax = df_.set_index('date').plot()

    pred_uc_m.plot(ax=ax, label='forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.5)

    ax.set_xlabel('Date')
    ax.set_ylabel('Transaction')
    plt.legend()

    # plt.show()

    # Create directory
    dir_name = 'res/' + datetime.now().strftime("%Y-%m-%d")

    # Create target Directory if don't exist
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("Directory ", dir_name, " Created ")
    else:
        print("Directory ", dir_name, " already exists")

    plt.savefig(dir_name + '/' +str(datetime.now().strftime("%H:%M")))
