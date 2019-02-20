from kisimulator.reward.generator import *
from kisimulator.reward.rewarder import *

res = generate_data()
compute_reward_transaction(res)

plt.show()