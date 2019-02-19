import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import os
from dotenv import load_dotenv
load_dotenv()


from datetime import timedelta
from datetime import datetime
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
np.random.seed = 1

# Data generator config
start_date = os.getenv("start_date")
number_of_days = os.getenv("number_of_days")
end_date = datetime.strptime(start_date, "%d/%m/%Y") + timedelta(days=number_of_days)
time_per_block = os.getenv("time_per_block")
sampling_freq = os.getenv("")
block_per_time_unit = round(3600 / time_per_block) if sampling_freq == 'H' else round(86400 / time_per_block)
tx_ampl_param = os.getenv("")
# trend = 'arctan'
# trend = 'linear'

# Trend [ "data" | "gen" ]
trend = os.getenv("")

# Trend config
# arctan
amp = os.getenv("amp")
skw = os.getenv("skw")
loc = os.getenv("loc")

# linear
a = os.getenv("a")
b = os.getenv("b")

# Plotting config
plot_tx_per_hour = os.getenv("plot_tx_per_hour")
plot_tx_per_blocks = os.getenv("plot_tx_per_blocks")
plot_epoch_per_blocks = os.getenv("plot_epoch_per_blocks")

# Logging config (in-console)
log_tx_per_hour = os.getenv("log_tx_per_hour")
log_tx_per_blocks = os.getenv("log_tx_per_blocks")

# Epochs config
# epochs = {1: 1., 2: 1., 3: 1., 4: 1.}
# epochs = {1: 0., 2: 0.3, 3: 0.7, 4: 0.9}
# thresholds = {1: 0.8, 2: 0.2, 3: 0.1}

epochs = {1: 0., 2: 1.}
thresholds = {1: 0.5, 2: -10, 3: -10}

# Data source [ "btc" | "eth" | "ark"]
data_ = "btc"


def generate_blocks():
    print('Generating blocks')

    # Trend from data
    if trend == 'data':
        data_file = 'data/' + data_ + '/n-transactions-all.csv'
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


def generate_validators():
    return dict


def distribute_validators(dict_of_validators):
    return


# if __name__ == '__main__':
#     print('Carbonara is the most awesome pizza.')

    # res = generate_blocks()
    # compute_reward_block(res)
    # compute_reward_transaction(res)
    # compute_reward_hybrid(res, 0.1)
