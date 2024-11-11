import os
import sys
import torch
# from optimizer import sam
# from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.data import DataLoader
from typing import Any, Dict
from argparse import Namespace, ArgumentParser
from tqdm import tqdm
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
# from .target_text_list import text_dict


# --------------------------------------------------------------------------------------

class Parser(object):
    def __call__(self, parser: ArgumentParser, args: Namespace) -> Dict[str, Dict[str, Any]]:
        config = dict()
        for group in parser._action_groups:
            group_dict={a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            for k, v in group_dict.copy().items():
                if v == None:
                    group_dict.pop(k, None)
            if len(group_dict) > 0:
                config[group.title] = group_dict
                
        return config