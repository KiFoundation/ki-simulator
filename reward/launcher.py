from reward.generator import *
from reward.rewarder import *
from reward.predictor import *

res = generate_data()
# print(res)
compute_reward_transfer(res, 0.2)


# res = generate_transactions_per_time_unit()
# forecasting(res)

plt.show()
