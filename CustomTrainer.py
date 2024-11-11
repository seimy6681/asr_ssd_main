import pandas as pd
import debugpy
import jiwer
# import torchaudio
import random
from transformers import Trainer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report
import pandas as pd
from tqdm import tqdm
import wandb
from typing import Any, Tuple, Union, List, Callable, Optional
from features.target_text_dict import target_text_dict

# HARD CODING WARNING: IF USING LOSS_FEATURE AS SELECT_DIFFICULT_WORDS, PLEASE PRE DEFINE THE SELECTED WORD LIST HERE BEFORE TRAINING:
SELECTED_DIFFICULT_WORDS = ['색종이', '화장실', '머리', '호랑이','컵','사탕']
SELECTED_DIFFICULT_WORDS = [target_text_dict.get(word, None) for word in SELECTED_DIFFICULT_WORDS]

class CustomTrainer(Trainer):
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config  # Save config as an instance attribute
        
    def compute_loss(self, model, inputs, return_outputs=False):

        if self.config.loss_feature == 'word_cer':
            loss_feature = inputs.pop(self.config.loss_feature)
            outputs = model(**inputs)
            base_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            weights = 1 + loss_feature.to(base_loss.device)
            weighted_loss = base_loss * weights.view(-1)
            
            return (weighted_loss.mean(), outputs) if return_outputs else weighted_loss.mean()
    
        if self.config.loss_feature == 'target_text_id':
            loss_feature = inputs.pop(self.config.loss_feature)
            # check if the current target word is in the custom difficult level word list
            outputs = model(**inputs)
            # logits = outputs.logits
            # labels = inputs["labels"]
            # log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # ctc_loss_fn = nn.CTCLoss(blank=model.config.pad_token_id, reduction='none',zero_infinity=True)
            # input_lengths = torch.full(size=(logits.size(0),), fill_value=logits.size(1), dtype=torch.long)
            # label_lengths = torch.full(size=(labels.size(0),), fill_value=labels.size(1), dtype=torch.long)
            
            # losses = ctc_loss_fn(log_probs.transpose(0,1), labels, input_lengths, label_lengths)
            
            base_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
            difficult_words = torch.tensor(SELECTED_DIFFICULT_WORDS, device=base_loss.device) # Convert diff.word list into tensor for comparsion
            mask = torch.isin(loss_feature, difficult_words)
            
            epsilon = 10**(-8)
            dynamic_scaling_factor = 1 + (base_loss / (base_loss + epsilon)) * 1.5
            
            weighted_loss = torch.where(mask, base_loss*1.5, base_loss)
            # weighted_loss = base_loss ** 2 # square the loss for the difficult level word
            # print('hi')
        else:
            outputs = model(**inputs)
            weighted_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            # weighted_loss = base_loss # trivial
            
        return (weighted_loss.mean(), outputs) if return_outputs else weighted_loss.mean()
            