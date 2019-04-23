import numpy as np
from itertools import permutations
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt


def select(validators_dict):
    total_weight = sum(list(validators_dict.values()))
    th = np.random.randint(total_weight)

    keys = list(validators_dict.keys())
    np.random.shuffle(keys)
    for validator in keys:
        th -= validators_dict[validator]
        if th <= 0:
            return validator

    return 0


def reputation(validator, behaviour_w, stake_w):
    rep = behaviour_w * validator[0] + stake_w * validator[1]
    return rep


validators = {}
validators_rep = {}
selected_num = {}
lst = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

validators_id = [''.join(i) for i in list(permutations(lst, 4))]

for id in validators_id[:200]:
    behavior_ = np.random.randint(100)
    stake_ = np.random.randint(100)
    validators[id] = [behavior_, stake_]
    selected_num[id] = [0, 0]
    validators_rep[id] = reputation(validators[id], 0.2, 0.8)

for j in range(80000):
    selected_ = select(validators_rep)
    selected_num[selected_][0] += 1
    selected_num[selected_][1] = validators_rep[selected_]

# normalize to compare trends
max_sel = max([i[0] for i in list(selected_num.values())])
max_rep = max([i[1] for i in list(selected_num.values())])

for item in selected_num:
    selected_num[item][0] /= max_sel
    selected_num[item][1] /= max_rep

max_beh = max([i[0] for i in list(validators.values())])
max_sta = max([i[1] for i in list(validators.values())])

for item in selected_num:
    validators[item][0] /= max_beh
    validators[item][1] /= max_sta

fig = plt.figure()
# fig.subplots_adjust(hspace=0.4, wspace=0.4)

# fig.add_subplot(2, 2, 1)
sns.distplot([i[0] for i in list(selected_num.values())], hist=False, label='selection')
# plt.xticks(rotation=90)
# plt.legend(loc='upper right')

# fig.add_subplot(2, 2, 2)
sns.distplot([i[1] for i in list(selected_num.values())], hist=False, label='reputation')
# plt.xticks(rotation=90)
# plt.legend(loc='upper right')

# fig.add_subplot(2, 2, 3)
sns.distplot([validators[i][1] for i in list(selected_num.keys())], hist=False, label='stake')
# plt.xticks(rotation=90)
# plt.legend(loc='upper right')

# fig.add_subplot(2, 2, 4)
sns.distplot([validators[i][0] for i in list(selected_num.keys())], hist=False, label='behaviour')
plt.xticks(rotation=90)
plt.legend(loc='upper right')

plt.show()
