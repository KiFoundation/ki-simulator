from utils.utils import *
from election.src.generator import *
from election.src.attacks import *

res_folder = create_res_forlder('election')
validators = generate_validators()
att_monopole(validators, res_folder)
