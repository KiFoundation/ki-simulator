import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dotenv import load_dotenv

# Load dotenv file
load_dotenv()

# Reward config
inflation = os.getenv("inflation")
base_reward = os.getenv("base_reward")

epochs = {1: 0., 2: 1.}
thresholds = {1: 0.5, 2: -10, 3: -10}


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

