import numpy as np
import pandas as pd
from itertools import permutations
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt

# Init and config e

selected_num = {}

plot_dist = False
plot_sec = True


# Generate validators
def generate_validators():
    # Create the validator IDs
    lst = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    validators_id = [''.join(i) for i in list(permutations(lst, 8))]

    return validators_id


# Create the validator dict {ID: [Behavior, Stake]} and validator rep dict {ID: rep}
# TODO: fix stake, num_validators_for_same_wallet and operator id. Should not be in this fucntion
def fill_reputation(validators_id_, weight_stake_, operator_id_, stake_op, seed=None):
    validators = {}
    validators_rep = {}

    np.random.seed(seed)

    # Generate the reputation scores for validators not belonging to the same operator
    total_stake = 0
    total_beh = 0
    for id in validators_id_:
        if id[0] != operator_id_:
            behavior_ = np.random.randint(100)
            stake_ = np.random.randint(100)
            total_stake += stake_
            total_beh += behavior_
            validators[id] = [50, stake_]
            validators_rep[id] = reputation(validators[id], 1 - weight_stake_, weight_stake_)

    # Generate the reputation scores for validators belonging to the same operator
    # Divide the stake of the operator over all of their validators equally

    stake_per_del = stake_op * total_stake / (1 - stake_op)
    stake_per_del = stake_per_del / (len(validators_id_) - len(validators_rep))
    beh_per_del = total_beh / (len(validators_id_) - len(validators_rep))
    beh_per_del = beh_per_del + 0.2 * beh_per_del
    print(beh_per_del)

    for id in validators_id_:
        if id[0] == operator_id_:
            # TODO: change this
            # Assume above average behavior score for operator's validators
            validators[id] = [50, stake_per_del]
            validators_rep[id] = reputation(validators[id], 1 - weight_stake_, weight_stake_)

    np.random.seed(None)

    return validators, validators_rep


# Reputation score :linear combination of the behaviour and stake metrics
def reputation(validator, behaviour_w, stake_w):
    rep = behaviour_w * validator[0] + stake_w * validator[1]
    return rep


# Sample a validator population with constraints on the number of validator per operator
def controled_sample(validators_id_, num_validators_for_same_wallet, num_validators_all, operator_id_, stake=0):
    # Validators for the same wallet :
    validators_for_same_wallet = []
    # Validators other :
    validators_for_others = []

    # Alter the validator distribution
    count = 0
    count_all = 0

    for validator in validators_id_:
        if count_all < num_validators_all - num_validators_for_same_wallet and validator[0] != operator_id_:
            validators_for_others.append(validator)
            count_all += 1
        if count < num_validators_for_same_wallet and validator[0] == operator_id_:
            validators_for_same_wallet.append(validator)
            count += 1

    validators_id_altered = validators_for_same_wallet + validators_for_others

    return validators_id_altered


# Weighted random selection function
def select_validator_per_weight(validators_dict):
    total_weight = sum(list(validators_dict.values()))
    th = np.random.randint(total_weight)

    keys = list(validators_dict.keys())
    np.random.shuffle(keys)
    for validator in keys:
        th -= validators_dict[validator]
        if th <= 0:
            return validator

    return 0


# Select validators for the round
def select_validators_rounds(validators_rep_, num_rounds_, num_validators_per_round_):
    selected_validators_ = []
    for j in range(num_rounds_):
        tmp_val_list = []
        for i in range(num_validators_per_round_):
            selected_ = select_validator_per_weight(validators_rep_)
            tmp_val_list.append(selected_)
        selected_validators_.append(tmp_val_list)

    return selected_validators_


# Normalize to compare trends TODO: recheck
def normalize(validators_, selected_num_):
    selected_num = {}
    val = {}

    max_sel = max([i[0] for i in list(selected_num_.values())])
    max_rep = max([i[1] for i in list(selected_num_.values())])

    for item in selected_num:
        selected_num[item][0] = selected_num_[item][0] / max_sel
        selected_num[item][1] = selected_num_[item][1] / max_rep

    max_beh = max([i[0] for i in list(validators_.values())])
    max_sta = max([i[1] for i in list(validators_.values())])

    for item in selected_num:
        val[item][0] = validators_[item][0] / max_beh
        val[item][1] = validators_[item][1] / max_sta

    return val


# Plot the distribution of the selection / Behaviour / Stake / Reputation
def plot_distributions(pd_selected_num_):
    fig = plt.figure()

    # fig.subplots_adjust(hspace=0.4, wspace=0.4)

    # fig.add_subplot(2, 2, 1)
    sns.distplot([i[0] for i in list(pd_selected_num_.values())], hist=False, label='selection')
    # plt.xticks(rotation=90)
    # plt.legend(loc='upper right')

    # fig.add_subplot(2, 2, 2)
    sns.distplot([i[1] for i in list(pd_selected_num_.values())], hist=False, label='reputation')
    # plt.xticks(rotation=90)
    # plt.legend(loc='upper right')

    # fig.add_subplot(2, 2, 3)
    sns.distplot([val[i][1] for i in list(pd_selected_num_.keys())], hist=False, label='stake')
    # plt.xticks(rotation=90)
    # plt.legend(loc='upper right')

    # fig.add_subplot(2, 2, 4)
    sns.distplot([val[i][0] for i in list(pd_selected_num_.keys())], hist=False, label='behaviour')
    plt.xticks(rotation=90)
    plt.legend(loc='upper right')

    plt.show()


# Count the number of a
def count_validator_per_operator(cvpo_selected_validators__round_, operator_id_):
    cvpo_count_ = 0
    for cvpo_val_ in cvpo_selected_validators__round_:

        if cvpo_val_[0] == operator_id_:
            cvpo_count_ += 1
    return cvpo_count_


Vaa = generate_validators()

# generate Y validators with X belonging to the same person
# for R runs select V validators
# measure A = X'/V where X' = intersection (X, V)
# avg A for R
# store Max (A)
# repeat for values of X
# repeat fot values of Y

weight_stake = 0.2

stake_operator = 0.01
num_rounds = 1
num_validators_per_round = 51

Xs = np.arange(10, 200, 10)
Ys = np.arange(10, 500, 50)

df = pd.DataFrame(index=Xs, columns=Ys)
df_v = pd.DataFrame(index=Xs, columns=Ys)

for X in Xs:
    for Y in Ys:
        val_sampled = controled_sample(Vaa, X, Y, 'a')
        val, val_rep = fill_reputation(val_sampled, weight_stake, 'a', stake_operator, 1)
        tmp_A_ = []
        for R in range(10):
            tmp_round_val = select_validators_rounds(val_rep, num_rounds, num_validators_per_round)[0]
            tmp_count_op = count_validator_per_operator(tmp_round_val, 'a')
            tmp_count_op /= num_validators_per_round
            tmp_A_.append(tmp_count_op)
        df[Y][X] = np.mean(tmp_A_)
        df_v[Y][X] = np.sqrt(np.var(tmp_A_))

print(df)
print(df_v)

df.fillna(value=np.nan, inplace=True)
ax = plt.axes()
sns.heatmap(df, vmin=0, vmax=1, annot=True, ax=ax)
ax.set_title('Collusion risk for VPR = ' + str(num_validators_per_round))
ax.set_ylabel('Num of eligible validators for one operator')
ax.set_xlabel('Num of eligible validators')
plt.savefig("../../election/res/res-" + str(num_validators_per_round) + str(stake_operator) + str(weight_stake) + ".png")
# plt.show()
