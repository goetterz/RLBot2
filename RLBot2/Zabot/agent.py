import math
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from continuous_act import ContinuousAction
import your_obs
from continuous_policy import ContinuousPolicy

# You can get the OBS size from the rlgym-ppo console print-outs when you start your bot
OBS_SIZE = 89

# If you haven't set these, they are [256, 256, 256] by default
POLICY_LAYER_SIZES = [1024, 1024, 768, 512, 256]

class Agent:
	def __init__(self):
		self.action_parser = ContinuousAction()
		self.num_actions = len(self.action_parser._lookup_table)
		cur_dir = os.path.dirname(os.path.realpath(__file__))
		
		device = torch.device("cpu")
		self.policy = ContinuousPolicy(
                OBS_SIZE, # I think this is 89 by default
                16, # 8 * 2 = 16
                [1024, 1024, 768, 512, 256], # you seem to know these
                device, # this should be cpu for RLBot
                var_min=continuous_var_range[0.1], # 0.1 by default
                var_max=continuous_var_range[1.0], # 1.0 by default
            )
		self.policy.load_state_dict(torch.load(os.path.join(cur_dir, "model.pt"), map_location=device))
		torch.set_num_threads(1)

	def act(self, state):
    with torch.no_grad():
        actions, _ = self.policy.get_action(state, deterministic=True)
    actions = actions.reshape((-1, 8))
        actions[..., :5] = actions[..., :5].clip(-1, 1)
        # The final 3 actions handle are jump, boost and handbrake. They are inherently discrete so we convert them to either 0 or 1.
        actions[..., 5:] = actions[..., 5:] > 0
    return action