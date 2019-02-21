import os
import sys
import random
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
np.random.seed = os.environ["random_seed"]

# Data generator config
data_source = os.environ["data_source"]
trend = os.environ["trend"]
start_date = os.environ["start_date"]
sampling_freq = os.environ["sampling_freq"]
tx_ampl_param = int(os.environ["tx_ampl_param"])
number_of_days = int(os.environ["number_of_days"])
time_per_block = int(os.environ["time_per_block"])

# Derived config
end_date = datetime.strptime(start_date, "%d/%m/%Y") + timedelta(days=number_of_days)
block_per_time_unit = round(3600 / time_per_block) if sampling_freq == 'H' else round(86400 / time_per_block)

# Trend config
# arctan
amp = float(os.environ["amp"])
skw = float(os.environ["skw"])
loc = float(os.environ["loc"])

# linear
a = float(os.environ["a"])
b = float(os.environ["b"])

# Plotting config
plot_tx_per_time_unit = os.environ["plot_tx_per_time_unit"]
plot_tx_per_block = os.environ["plot_tx_per_block"]
plot_epoch_per_block = os.environ["plot_epoch_per_block"]

# Logging config (in-console)
log_tx_per_time_unit = int(os.environ["log_tx_per_time_unit"])
log_tx_per_block = int(os.environ["log_tx_per_block"])

# Epochs config # TODO : ADD EPOCH CONFIG TO THE .ENV FILE
# epochs = {1: 1., 2: 1., 3: 1., 4: 1.}
epochs = {1: 0., 2: 0.3, 3: 0.7, 4: 0.9}
thresholds = {1: 0.8, 2: 0.2, 3: 0.1}
# epochs = {1: 0., 2: 1.}
# thresholds = {1: 0.5, 2: -10, 3: -10}

# Validator config
num_of_active_validators = 20

# Plotting settings
subplots = [plot_tx_per_time_unit, plot_tx_per_block, plot_epoch_per_block].count('True')
subplots_pos = 1


def generate_transactions_per_time_unit():
    # Trend from data
    if trend == 'data':
        print('Loading {} data from file'.format(data_source))

        # Load the data from the file
        data_file = 'data/' + data_source + '/n-transactions-all.csv'
        df = pd.read_csv(data_file, names=['date', 'data'])

        # Set type to timestamp
        df['date'] = pd.to_datetime(df['date'])
        print(df)

        # Drop rows until the start date
        df = df.drop(df.index[:len(df.loc[df.date < start_date])]).head(number_of_days)
        df['data'] /= tx_ampl_param
        print(df)
        print("transaction distribution loaded")
        sys.stdout.flush()

    else:
        print('Generating transactions per time unit')

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

    if plot_tx_per_time_unit == "True":
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

    print("Transactions distributed over blocks")
    sys.stdout.flush()

    if log_tx_per_block != 0:
        log_transactions(df_blocks, log_tx_per_block)

    if plot_tx_per_block == "True":
        plot_transactions(df_blocks, "tpb")

    return df_blocks


def set_epochs(transactions_per_block):
    df_blocks = transactions_per_block.copy()

    # Set epochs
    df_blocks['epoch'] = [-1 for i in range(len(df_blocks))]
    for i in range(int(len(df_blocks) / num_of_active_validators) + 1):
        empty_block_ratio = 1 - np.count_nonzero(df_blocks['tx'][i * num_of_active_validators:i * num_of_active_validators + num_of_active_validators]) / num_of_active_validators

        if empty_block_ratio > thresholds[1]:
            for j in range(num_of_active_validators):
                df_blocks.at[i * num_of_active_validators + j, 'epoch'] = 1
        if thresholds[2] < empty_block_ratio <= thresholds[1]:
            for j in range(num_of_active_validators):
                df_blocks.at[i * num_of_active_validators + j, 'epoch'] = 2
        if thresholds[3] < empty_block_ratio <= thresholds[2]:
            for j in range(num_of_active_validators):
                df_blocks.at[i * num_of_active_validators + j, 'epoch'] = 3
        if empty_block_ratio <= thresholds[3]:
            for j in range(num_of_active_validators):
                df_blocks.at[i * num_of_active_validators + j, 'epoch'] = 4

    # Drop the added lines ( +1)
    df_blocks = df_blocks.dropna(0, how='any')

    print("transaction epochs set")
    sys.stdout.flush()

    if plot_epoch_per_block == "True":
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
    global subplots_pos

    if what == "tptu":
        plt.subplot(subplots, 1, subplots_pos)
        sns.lineplot(df.index.values, df['data'])
        plt.xlabel('time unit (' + sampling_freq + ")")
        plt.ylabel('tx')
        plt.title('Transactions per time unit')
        plt.grid(True)
        subplots_pos += 1

    if what == "tpb":
        plt.subplot(subplots, 1, subplots_pos)
        sns.lineplot(df.index.values, df['tx'])
        plt.xlabel('block')
        plt.ylabel('tx')
        plt.title('Transactions per block')
        plt.grid(True)
        subplots_pos += 1

    if what == "epoch":
        plt.subplot(subplots, 1, subplots_pos)
        sns.scatterplot(df.index.values, df['tx'], hue=df['epoch'], legend="full", s=6, linewidth=0.0)
        plt.xlabel('block')
        plt.ylabel('tx')
        plt.title('Transactions per block')
        plt.grid(True)
        subplots_pos += 1

    plt.subplots_adjust(hspace=1)


def generate_validators(num_of_validators):
    validators = {}
    for i in range(num_of_validators):
        validators['val_' + str(i)] = []

    return validators


def distribute_validators(df_blocks, num_of_validators):
    validators = generate_validators(num_of_validators)
    val_rounds = round(len(df_blocks)/num_of_active_validators)

    df_validators = df_blocks.copy()

    # Set validators
    df_validators['validator'] = ["" for i in range(len(df_blocks))]

    for val_round in range(val_rounds):
        active_validators_ids = random.sample(list(validators), num_of_active_validators)
        random.shuffle(active_validators_ids)
        for j in range(num_of_active_validators):
            df_validators.at[val_round * num_of_active_validators + j, 'validator'] = active_validators_ids[j]

    # Drop the added lines ( +1)
    df_validators = df_validators.dropna(0, how='any')

    return df_validators


def generate_data():
    return distribute_validators(set_epochs(generate_blocks(generate_transactions_per_time_unit())),150)
