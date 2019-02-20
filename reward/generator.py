import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from datetime import timedelta
from datetime import datetime
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# Load dotenv file
load_dotenv()

# Set random stat
np.random.seed = os.getenv("random_seed")

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
plot_tx_per_time_unit = os.getenv("plot_tx_per_time_unit")
plot_tx_per_block = os.getenv("plot_tx_per_block")
plot_epoch_per_block = os.getenv("plot_epoch_per_block")

# Logging config (in-console)
log_tx_per_time_unit = os.getenv("log_tx_per_time_unit")
log_tx_per_block = os.getenv("log_tx_per_block")

# Epochs config
# epochs = {1: 1., 2: 1., 3: 1., 4: 1.}
# epochs = {1: 0., 2: 0.3, 3: 0.7, 4: 0.9}
# thresholds = {1: 0.8, 2: 0.2, 3: 0.1}
#  TODO : ADD EPOCH CONFIG TO THE .ENV FILE
epochs = {1: 0., 2: 1.}
thresholds = {1: 0.5, 2: -10, 3: -10}

# Data source [ "btc" | "eth" | "ark"]
data_ = "btc"


def generate_transactions_per_time_unit():
    print('Generating transactions per time unit : ', sampling_freq)

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

    if log_tx_per_time_unit != 0:
        log_transactions(df, log_tx_per_time_unit)

    if plot_tx_per_time_unit :
        plot_transactions(df, "tptu")

    return df


def generate_blocks(transactions_per_time_unit):
    print('Generating blocks')

    # Split the time units to block : 1 block every 8 seconds
    df_blocks = pd.DataFrame([0 for i in range(len(transactions_per_time_unit) * block_per_time_unit)], columns=['tx'])

    # Randomly split the tx/h of each timestamp over the 450 hourly blocks : use multinomial distribution (fixed sum)
    k = 0
    for index, row in transactions_per_time_unit.iterrows():

        txs = np.random.multinomial(row['data'], [1 / float(block_per_time_unit)] * block_per_time_unit, size=1)
        for tx in txs[0]:
            df_blocks.loc[k] = tx
            k += 1

    print("transactions distributed over blocks")
    sys.stdout.flush()

    if log_tx_per_block != 0:
        log_transactions(df_blocks, log_tx_per_block)

    if plot_tx_per_block :
        plot_transactions(df_blocks, "tpb")

    return df_blocks


def set_epochs(transactions_per_block):
    df_blocks = transactions_per_block.copy()

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

    if plot_epoch_per_block :
        plot_transactions(df_blocks, "epoch")

    return df_blocks


def log_transactions(df, lines):
    # Logging
    if lines > 0:
        print(df.head(lines))

    if lines < 0:
        print(df.tail(abs(lines)))


def plot_transactions(df, what):
    # Plotting
    sns.set(style="ticks")

    if what == "epoch":
        sns.scatterplot(df.index.values, df['epoch'])

    if what == "tptu":
        sns.lineplot(df.index.values, df['data'])

    if what == "tpb" :
        sns.lineplot(df.index.values, df['tx'])

    plt.show()


def generate_validators():
    return dict


def distribute_validators(dict_of_validators):
    return

