from kisimulator.reward.generator import *
from kisimulator.reward.rewarder import *
from kisimulator.reward.predictor import *

res = generate_data()
print(res)
compute_reward_transfer(res, 0.2)

# res = generate_transactions_per_time_unit()
# forecasting(res)

plt.show()