import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dotenv import load_dotenv

# Load dotenv file
load_dotenv()

# Reward config
inflation = float(os.getenv("inflation"))
base_reward = float(os.getenv("base_reward"))
max_tx_per_block = float(os.getenv("max_tx_per_block"))
time_per_block = int(os.environ["time_per_block"])
total_supply = int(os.environ["total_supply"])
num_of_blocks_per_time_unit = round(86400 / time_per_block)
num_of_blocks_per_year = 365 * num_of_blocks_per_time_unit
max_predictor = os.environ["max_predictor"]

epochs = {1: 0.7, 2: 0.8, 3: 0.9, 4: 1.}
thresholds = {1: 0.4, 2: 0.2, 3: 0.0}


def compute_reward_transaction(df_blocks_):
    df_blocks_rewards = pd.DataFrame([[0, 0., '', 0.] for i in range(len(df_blocks_))],
                                     columns=['tx', 'epoch', 'validator', 'reward'])

    for index, row in df_blocks_.iterrows():
        payed_rewards = sum(df_blocks_rewards['reward'][:index])
        remaining_transactions = sum(df_blocks_['tx'][index:])
        r_hat = base_reward * (1 - epochs[row['epoch']]) + epochs[row['epoch']] * (
                row['tx'] * (inflation - payed_rewards)) / remaining_transactions

        df_blocks_rewards.iloc[index] = [row['tx'], row['epoch'], row['validator'], r_hat]

    print("The total sum of payed reward is ", sum(df_blocks_rewards['reward']))

    return df_blocks_rewards


def compute_reward_block(df_blocks_):
    df_blocks_rewards = pd.DataFrame([[0, 0., '', 0.] for i in range(len(df_blocks_))],
                                     columns=['tx', 'epoch', 'validator', 'reward'])

    for index, row in df_blocks_.iterrows():
        payed_rewards = sum(df_blocks_rewards['reward'][:index])
        r_hat = base_reward * (1 - epochs[row['epoch']]) + epochs[row['epoch']] * (inflation - payed_rewards) / (
                len(df_blocks_rewards) - index)
        df_blocks_rewards.iloc[index] = [row['tx'], row['epoch'], row['validator'], r_hat]
        # print(inflation - payed_rewards, '\t', len(df_blocks_rewards) - index, '\t',
        #       (inflation - payed_rewards) / (len(df_blocks_rewards) - index), '\t', base_reward * (1 - epochs[row['epoch']]))

    print("The total sum of payed reward is ", sum(df_blocks_rewards['reward']))

    return df_blocks_rewards


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

    print("The total sum of payed reward is ", sum(df_blocks_rewards['reward']))

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

    return df_blocks_rewards


def compute_reward_transfer(df_blocks_, static_split_, i, itt):
    # Static inflation
    sc = static_split_ * inflation

    # Dynamic inflation
    dc = (1 - static_split_) * inflation

    # Static reward
    static_reward = sc / len(df_blocks_)

    # Theoretical dynamic reward : what should be payed if all block validated in the year are full
    theoretical_dynamic_rewards = dc / len(df_blocks_)

    # Transferable reward
    tr_alpha = 1.

    # Final data structure
    df_blocks_rewards = pd.DataFrame([[0, 0., 0, 0., 0] for i in range(len(df_blocks_))],
                                     columns=['tx', 'rewardT', 'epoch', 'reward', 'max'])
    df_blocks_rewards = df_blocks_rewards.set_index(df_blocks_.index.values)

    # The filling rate of a block : i.e. nb_of_tx / max_nb_of_tx
    filling_rate = 0

    # The dynamic reward for the current block
    dynamic_reward = 0

    # Filling rate of the previous block
    filling_rate_prev = 0

    # The total amount of transferred rewards
    transferred_reward = 0

    # Total amount of payed dynamic rewards
    payed_dynamic_rewards = 0

    # temp
    filling_rate_mean = 0
    transferred_reward1 = []

    print("sc : {}, dc : {}, static_reward : {}, theoretical_dynamic_rewards : {}".format(sc, dc, static_reward,
                                                                                          theoretical_dynamic_rewards))
    # it = i * len(df_blocks_)
    it = 0
    jt = 0

    yearly_jump = 400
    smoothing_blocks = 9000

    # For each block
    for index, row in df_blocks_.iterrows():
        global max_tx_per_block
        
        if len(df_blocks_.loc[index:]) - 1 != 0 and index % 144 == 0:
            it += 1

            # Static max
            if max_predictor == 'static':
                max_tx_per_block = 6000

            # Max prediction with amplifying factor amp_max
            if max_predictor == 'max':
                amp_max = 2
                max_tx_per_block = amp_max * max([max(df_blocks_.loc[index - 144:index]['tx']), 1])

            # Mean prediction with amplifying factor amp_mean
            if max_predictor == 'mean':
                amp_mean = 4
                max_tx_per_block = amp_mean * np.mean([max(df_blocks_.loc[index - 144:index]['tx']), 1])

            # Linear annual growth with yearly slop adjustment
            if max_predictor == 'lin-adj':
                max_tx_per_block = itt + (0.02 + 0.2 * i) * it

            # Steps
            if max_predictor == 'steps':
                max_tx_per_block = itt + 200 * i

            # Steps with smoothing
            if max_predictor == 'step-sm':
                if len(df_blocks_.loc[:index]) <= smoothing_blocks:
                    max_tx_per_block = yearly_jump * (i + 1 / (1 + np.exp(-0.1 * it)))
                    jt += 1

                if smoothing_blocks < len(df_blocks_.loc[:index]) < len(df_blocks_) - smoothing_blocks:
                    max_tx_per_block = yearly_jump * (i + 1)
                    jt += 1

                if len(df_blocks_) - smoothing_blocks <= len(df_blocks_.loc[:index]):
                    max_tx_per_block = yearly_jump * (1 + i + 1 / (1 + np.exp(-0.1 * (it - 365))))

            # print(itt)

        # Compute the filling rate
        filling_rate = min(row['tx'] / max_tx_per_block, 1)
        filling_rate_mean = filling_rate_mean + filling_rate

        # Compute the transferable reward : i.e. the reward for the unfilled portion of the block
        transferred_reward = transferred_reward + tr_alpha / len(df_blocks_.loc[index:]) * (
                1 - filling_rate_prev) * theoretical_dynamic_rewards

        transferred_reward1.append(
            tr_alpha / len(df_blocks_.loc[index:]) * (1 - filling_rate_prev) * theoretical_dynamic_rewards)

        # Compute The dynamic reward to pay for the block: sum of the reward for the filled part and transfered reward
        dynamic_reward = filling_rate * theoretical_dynamic_rewards + min(transferred_reward,
                                                                          theoretical_dynamic_rewards)
        # print(transferred_reward1[-10:], sum(transferred_reward1[-10:]))
        dynamic_reward1 = filling_rate * (
                theoretical_dynamic_rewards + min(transferred_reward, theoretical_dynamic_rewards))

        # Store variables for the next block
        payed_dynamic_rewards = payed_dynamic_rewards + dynamic_reward
        filling_rate_prev = filling_rate

        # Total reward : static + dynamic
        r_hat = static_reward + dynamic_reward
        r_hat1 = static_reward + dynamic_reward1

        # Fill the result structure
        df_blocks_rewards.loc[index] = [row['tx'], r_hat, row['epoch'], r_hat1, max_tx_per_block]

    print(jt)
    print("Objective reward amount to be payed is ", inflation,
          "\t\t Actual objective inflation rate is :", inflation / total_supply)

    print("The total sum of payed reward (trend) is ", sum(df_blocks_rewards['rewardT']),
          "\t\t Actual inflation rate is :", sum(df_blocks_rewards['rewardT']) / total_supply)

    print("The total sum of payed reward is ", sum(df_blocks_rewards['reward']),
          "\t\t Actual inflation rate is :", sum(df_blocks_rewards['reward']) / total_supply)

    # plt.subplot(3, 1, 1)
    # sns.lineplot(df_blocks_rewards.index.values, df_blocks_rewards['tx'])
    #
    # plt.subplot(3, 1, 2)
    # sns.lineplot(df_blocks_rewards.index.values, df_blocks_rewards['rewardT'])
    #
    # plt.subplot(3, 1, 3)
    # sns.lineplot(df_blocks_rewards.index.values, df_blocks_rewards['reward'])

    return df_blocks_rewards, max_tx_per_block


def compute_reward_transfer_multi_years(df_blocks_, static_split_):
    results = pd.DataFrame()
    itt = 100
    for i in range(0, int(len(df_blocks_) / num_of_blocks_per_year), 1):
        print("Year", i + 1)
        res, itt = compute_reward_transfer(
            df_blocks_[num_of_blocks_per_year * i: num_of_blocks_per_year * i + num_of_blocks_per_year], static_split_,
            i, itt)
        print(itt)
        results = results.append(res)

    this_file = os.path.abspath(os.path.dirname(__file__))
    res_file = os.path.join(this_file, '../res/results' + str(datetime.date.today()) + '.csv')
    results.to_csv(res_file)

    plt.subplot(3, 1, 1)
    sns.lineplot(results.index.values, results['tx'])

    plt.subplot(3, 1, 2)
    sns.lineplot(results.index.values, results['reward'])

    plt.subplot(3, 1, 3)
    sns.lineplot(results.index.values, results['max'])
