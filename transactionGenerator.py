import sys
import pandas as pd
import numpy as np
import itertools
from datetime import timedelta
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
np.random.seed = 1

# Data generator config
start_date = '2/11/2018'
number_of_days = 3000
end_date = datetime.strptime(start_date, "%d/%m/%Y") + timedelta(days=number_of_days)
time_per_block = 600
sampling_freq = 'D'
block_per_time_unit = round(3600 / time_per_block) if sampling_freq == 'H' else round(86400 / time_per_block)
tx_ampl_param = 10
# trend = 'arctan'
# trend = 'linear'
trend = 'data'

# Trend config
# arctan
amp = 1
skw = 3
loc = 1

# linear
a = 0.2
b = 0.

# Plotting config
plot_tx_per_hour = False
plot_tx_per_blocks = False
plot_epoch_per_blocks = False

# Logging config (in-console)
log_tx_per_hour = 0
log_tx_per_blocks = 0

# Epochs config
# epochs = {1: 1., 2: 1., 3: 1., 4: 1.}
# epochs = {1: 0., 2: 0.3, 3: 0.7, 4: 0.9}
# thresholds = {1: 0.8, 2: 0.2, 3: 0.1}

epochs = {1: 0., 2: 1.}
thresholds = {1: 0.5, 2: -10, 3: -10}

# Reward config
inflation = 100000
base_reward = 5.5

data_file = 'data/bitcoin/n-transactions-all.csv'
# data_file = 'data/ethereum/export-TxGrowth.csv'

def generate_blocks():
    # Trend from data
    if trend == 'data':
        df = pd.read_csv(data_file, names=['date', 'data'], nrows=number_of_days, skiprows=0)
        df['data'] /= tx_ampl_param

    else:
        # Create the timestamps
        date_rng = pd.date_range(start=start_date, end=end_date, freq=sampling_freq)

        # Create the dataframe holding the the tx/time unit
        df = pd.DataFrame(date_rng, columns=['date'])
        # df['data'] = np.random.normal(1, 0.1, size=(len(date_rng)))

        # Fill the dataframe with the genrated values:
        df['ind'] = df.index.values / float(len(df))

        # Arctan trend
        if trend == 'arctan':
            df['data'] = amp * (np.arctan(loc) + np.arctan(skw * df['ind'] - loc))

        # Linear trend
        if trend == 'linear':
            df['data'] = a * df['ind'] + b

        print("transaction distribution generated")
        sys.stdout.flush()

        # Round and scale the values
        df = df.round({'data': 3})
        df['data'] *= tx_ampl_param

    forecasting(df)

    # Split the time units to block : 1 block every 8 seconds
    df_blocks = pd.DataFrame([0 for i in range(len(df) * block_per_time_unit)], columns=['tx'])

    # Randomly split the tx/h of each timestamp over the 450 hourly blocks : use multinomial distribution (fixed sum)
    k = 0
    for index, row in df.iterrows():

        txs = np.random.multinomial(row['data'], [1 / float(block_per_time_unit)] * block_per_time_unit, size=1)
        for tx in txs[0]:
            df_blocks.loc[k] = tx
            k += 1

    print("transactions distributed over blocks")
    sys.stdout.flush()

    # Set epochs
    df_blocks['epoch'] = [-1 for i in range(len(df_blocks))]
    for i in range(int(len(df_blocks) / 51) + 1):
        empty_block_ratio = 1 - np.count_nonzero(df_blocks['tx'][i * 51:i * 51 + 51]) / 51

        if empty_block_ratio > thresholds[1]:
            for j in range(51):
                df_blocks.at[i * 51 + j, 'epoch'] = 1
        if thresholds[2] < empty_block_ratio <= thresholds[1]:
            for j in range(51):
                df_blocks.at[i * 51 + j, 'epoch'] = 2
        if thresholds[3] < empty_block_ratio <= thresholds[2]:
            for j in range(51):
                df_blocks.at[i * 51 + j, 'epoch'] = 3
        if empty_block_ratio <= thresholds[3]:
            for j in range(51):
                df_blocks.at[i * 51 + j, 'epoch'] = 4

    # Drop the added lines ( +1)
    df_blocks = df_blocks.dropna(0, how='any')

    print("transaction epochs set")
    sys.stdout.flush()

    # Logging
    if log_tx_per_blocks > 0:
        print(df_blocks.head(log_tx_per_blocks))

    if log_tx_per_blocks < 0:
        print(df_blocks.tail(abs(log_tx_per_blocks)))

    if log_tx_per_hour > 0:
        print(df.head(log_tx_per_hour))

    if log_tx_per_hour < 0:
        print(df.tail(abs(log_tx_per_hour)))

    # Plotting
    sns.set(style="ticks")

    if plot_epoch_per_blocks:
        sns.scatterplot(df_blocks.index.values, df_blocks['epoch'])

    if plot_tx_per_hour:
        # plt.xticks(rotation=90)
        sns.lineplot(df.index.values, df['data'])

    if plot_tx_per_blocks:
        sns.lineplot(df_blocks.index.values, df_blocks['tx'])

    if plot_tx_per_hour or plot_tx_per_blocks or plot_epoch_per_blocks:
        plt.show()

    return df_blocks


def compute_reward_transaction(df_blocks_):
    df_blocks_rewards = pd.DataFrame([[0, 0., 0] for i in range(len(df_blocks_))], columns=['tx', 'reward', 'epoch'])

    for index, row in df_blocks_.iterrows():
        payed_rewards = sum(df_blocks_rewards['reward'][:index])
        remaining_transactions = sum(df_blocks_['tx'][index:])
        r_hat = base_reward * (1 - epochs[row['epoch']]) + epochs[row['epoch']] * (
                row['tx'] * (inflation - payed_rewards)) / remaining_transactions

        df_blocks_rewards.iloc[index] = [row['tx'], r_hat, row['epoch']]

    print(sum(df_blocks_rewards['reward']))

    # compute mean reward per round
    mean_rewards_per_round = []
    mean_rewards_per_round_x = []
    mean_rewards_per_round_epoch = []
    for i in range(int(len(df_blocks_rewards) / 51) + 1):
        mean_reward_per_round = np.mean(df_blocks_rewards['reward'][i * 51:i * 51 + 51])
        mean_rewards_per_round.append(mean_reward_per_round)
        mean_rewards_per_round_x.append(i)
        mean_rewards_per_round_epoch.append(df_blocks_rewards.iloc[i * 51]['epoch'])

    # sample the dataframe
    # df_blocks_rewards_sample = df_blocks_rewards.sample(frac=0.1, replace=True, random_state=1)

    sns.scatterplot(mean_rewards_per_round_x, mean_rewards_per_round, hue=mean_rewards_per_round_epoch, s=10)
    plt.show()
    return


def compute_reward_block(df_blocks_):
    df_blocks_rewards = pd.DataFrame([[0, 0.] for i in range(len(df_blocks_))], columns=['tx', 'reward'])

    for index, row in df_blocks_.iterrows():
        payed_rewards = sum(df_blocks_rewards['reward'][:index])
        r_hat = base_reward * (1 - epochs[row['epoch']]) + epochs[row['epoch']] * (inflation - payed_rewards) / (
                len(df_blocks_rewards) - index)
        df_blocks_rewards.iloc[index] = [row['tx'], r_hat]
        # print(inflation - payed_rewards, '\t', len(df_blocks_rewards) - index, '\t',
        #       (inflation - payed_rewards) / (len(df_blocks_rewards) - index), '\t', base_reward * (1 - epochs[row['epoch']]))

    print(sum(df_blocks_rewards['reward']))

    # compute mean reward per round
    mean_rewards_per_round = []
    mean_rewards_per_round_x = []
    for i in range(int(len(df_blocks_rewards) / 51) + 1):
        mean_reward_per_round = np.mean(df_blocks_rewards['reward'][i * 51:i * 51 + 51])
        mean_rewards_per_round.append(mean_reward_per_round)
        mean_rewards_per_round_x.append(i)

    # sample the dataframe
    # df_blocks_rewards_sample = df_blocks_rewards.sample(frac=0.25, replace=True, random_state=1)

    sns.scatterplot(mean_rewards_per_round_x, mean_rewards_per_round, s=10)
    plt.show()
    return


def compute_reward_hybrid(df_blocks_, static_split_):
    sc = static_split_ * inflation
    dc = (1 - static_split_) * inflation

    df_blocks_rewards = pd.DataFrame([[0, 0., 0] for i in range(len(df_blocks_))], columns=['tx', 'reward', 'epoch'])

    payed_dynamic_rewards = 0
    for index, row in df_blocks_.iterrows():
        if index < 1:
            # dynamic_reward = epochs[row['epoch']] * (row['tx'] * (dc - payed_dynamic_rewards)) / (
            #             len(df_blocks_) * 2)
            dynamic_reward = 2

        else:
            # print((len(df_blocks_rewards) - index) * max([np.mean(df_blocks_['tx'][index-1:index]), 1]), sum(df_blocks_['tx'][index:]))

            dynamic_reward = epochs[row['epoch']] * (row['tx'] * (dc - payed_dynamic_rewards)) / (
                    (len(df_blocks_rewards) - index) * max([np.mean(df_blocks_['tx'][:index]), 1]))

        static_reward = (1 - epochs[row['epoch']]) * sc / len(df_blocks_)
        payed_dynamic_rewards = payed_dynamic_rewards + dynamic_reward

        r_hat = static_reward + dynamic_reward
        df_blocks_rewards.iloc[index] = [row['tx'], r_hat, row['epoch']]

    print(sum(df_blocks_rewards['reward']))

    # compute mean reward per round
    mean_rewards_per_round = []
    mean_rewards_per_round_x = []
    mean_rewards_per_round_epoch = []
    mean_tx_per_round = []
    round_ = 51
    for i in range(int(len(df_blocks_rewards) / round_) + 1):
        mean_reward_per_round = np.mean(df_blocks_rewards['reward'][i * round_:i * round_ + round_])
        mean_tx_per_round.append(np.mean(df_blocks_rewards['tx'][i * round_:i * round_ + round_]))
        mean_rewards_per_round.append(mean_reward_per_round)
        mean_rewards_per_round_x.append(i)
        # mean_rewards_per_round_epoch.append(df_blocks_rewards.iloc[i * round_]['epoch'])

    # sample the dataframe
    # df_blocks_rewards_sample = df_blocks_rewards.sample(frac=0.1, replace=True, random_state=1)

    ax = sns.scatterplot(mean_rewards_per_round_x, mean_rewards_per_round, s=10, linewidth=0.0)
    ax2 = ax.twinx()
    sns.lineplot(mean_rewards_per_round_x, mean_tx_per_round, ax=ax2)
    plt.show()

    return


def generate_validators():
    return dict


def distribute_validators(dict_of_validators):
    return


def forecasting(df_blocks_):
    df_blocks_['date'] = pd.to_datetime(df_blocks_['date'])
    df_blocks_ = df_blocks_.set_index('date')
    df_blocks_ = df_blocks_['data'].resample('MS').mean()
    print(len(df_blocks_))

    decomposition = sm.tsa.seasonal_decompose(df_blocks_, model='multiplicative')
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

    mod = sm.tsa.statespace.SARIMAX(df_blocks_, order=a[0], seasonal_order=a[1],
                                    enforce_stationarity=False, enforce_invertibility=False)
    # Fit the selection model
    results = mod.fit()

    # Plot diagnostics
    results.plot_diagnostics(figsize=(16, 8))

    # Get and plot predictions staring a given date
    pred = results.get_prediction(start=pd.to_datetime('2015-01-01'), dynamic=False)
    pred_ci = pred.conf_int()

    df_ = pd.DataFrame({'date': df_blocks_['2014':].index.values,
                        'data': [df_blocks_['2014':][i] for i in df_blocks_['2014':].index.values]})

    ax = df_.set_index('date').plot(label='observed')

    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    plt.legend()

    # Compute MSE and RMSE error
    y_forecasted = pred.predicted_mean
    y_truth = df_blocks_['2017-01-01':]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    pred_uc = results.get_forecast(steps=100)
    pred_ci = pred_uc.conf_int()
    ax = df_blocks_.plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Furniture Sales')
    plt.legend()

    plt.show()

res = generate_blocks()
# compute_reward_block(res)
# compute_reward_transaction(res)
# compute_reward_hybrid(res, 0.1)
