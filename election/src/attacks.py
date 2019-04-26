import os
import pandas as pd
from election.src.generator import *

from dotenv import load_dotenv

# Load dotenv file
load_dotenv()

# Load config
weight_stake = float(os.getenv("weight_stake"))
stake_operator = float(os.getenv("stake_operator"))
num_rounds = int(os.getenv("num_rounds"))
num_validators_per_round = int(os.getenv("num_validators_per_round"))

Xs_conf = list(map(int, os.getenv("eligible_validators_for_one_operator_range").strip(' ').split(',')))
Ys_conf = list(map(int, os.getenv("eligible_validators_range").strip(' ').split(',')))

Xs = np.arange(Xs_conf[0], Xs_conf[1], Xs_conf[2])
Ys = np.arange(Ys_conf[0], Ys_conf[1], Ys_conf[2])


def att_monopole(validators_list, res_folder):
    # From the validator list sample Y validators with X belonging to the same person
    # Select V validators R times
    # measure A = X'/V where X' = intersection (X, V)
    # avg A for R
    # store Max (A)
    # repeat for values of X
    # repeat fot values of Y

    df = pd.DataFrame(index=Xs, columns=Ys)
    df_v = pd.DataFrame(index=Xs, columns=Ys)

    for X in Xs:
        for Y in Ys:
            val_sampled = controled_sample(validators_list, X, Y, 'a')
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
    plt.savefig(res_folder + '-'.join([str(num_validators_per_round), str(stake_operator), str(weight_stake)]) + ".pdf")
