from utils.utils import *
from reward.src.generator import *
from reward.src.rewarder import *
from reward.src.predictor import *


res_folder = create_res_forlder('reward')
res = generate_data()
print(res)

compute_reward_transfer_multi_years(res, 0.2)
# res = generate_transactions_per_time_unit()
# forecasting(res)

plt.show()
