import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchaudio.transforms as T
from transformers import AutoFeatureExtractor
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
import os
import math
import random
import jiwer
from util import Parser
import argparse, textwrap
from argparse import Namespace
import wandb
from typing import Tuple, Union
import debugpy
from train_asr import train_asr

torch.cuda.empty_cache()


DATA_PATH="/home/selinawisco/hdd/korean_asr"
WORKSPACE_PATH="/home/selinawisco/selina_main/asr/sel_wav2vec2_asr_main"

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DEBUGGING ------------------------------------------------------------
debugpy.listen(5678)
print("waiting for debugger")
debugpy.wait_for_client()
debugpy.breakpoint()
print('break on this line')
# ----------------------------------------------------------------------

#============================================
# arguments parsing
#============================================
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
default_group = parser.add_argument_group('default')
default_group.add_argument('--train', default=True, help='train mode')
default_group.add_argument('--num_runs', type=int, default=1, help='number of consecutive runs with consecutive random seeds')
default_group.add_argument('--eval_mode', default=False, action="store_true", help='model test mode')
default_group.add_argument('--checkpoint', type=str, help='model checkpoint to test')
default_group.add_argument('--verbose', action="store_true")
default_group.add_argument('--data_path', type=str, default=DATA_PATH, help="directory path of data files")
default_group.add_argument('--task', type=str, default='asr_finetuning')
default_group.add_argument('--seed', type=int, default=42, help="random seed")
default_group.add_argument('--test_best_model', action="store_true")
default_group.add_argument('--save_name', type=str)
default_group.add_argument('--result_csv', action="store_true")
default_group.add_argument('--debug', action="store_true")

# wandb
wandb_group = parser.add_argument_group('wandb')
wandb_group.add_argument('--run_name', type=str, default="test", help="wandb run name")
wandb_group.add_argument('--logging_steps', type=int, default=50, help='wandb log & watch steps')
wandb_group.add_argument('--watch', type=str, default='all', help="wandb.watch parameter log")

# train args
train_group = parser.add_argument_group('train')
train_group.add_argument('--target', default='human_text_jamo', type=str, help="name of the target(column name in csv) for the custom loss function")
train_group.add_argument('--loss_feature', default=None, type=str, help="name of the feature(column name in csv) for the custom loss function")
train_group.add_argument('--asr_mode',type=str, default='human', help='human : transcribing natural pronunciation, target : transcribing target text')
train_group.add_argument('--dropout',type=str, help='ex. --dropout = "0.1:6 7 8" -> dropout_rate=0.1, layers_to_apply = [6,7,8]')


train_group.add_argument('--k_fold', type=int, help="k for k fold cross validation")
# train_group.add_argument('--callback', action='extend', nargs='*', type=str, 
#                          help=textwrap.dedent("""\
#                              callback list to use during training
#                              - es: early stopping
#                              - best: saving best model (watching vlidation loss)"""))
train_group.add_argument('--batch_size', type=int, default=8, help='batch size of training')
train_group.add_argument('--epochs', type=int, default=30, help='epochs of training')


# file 
file_group = parser.add_argument_group('file')
file_group.add_argument('--data_filename', type=str, default='hospital_target_all.csv')
file_group.add_argument('--splitted_data_files', default=False, action='store_true')
file_group.add_argument('--filter_dataset', default=False, action='store_true')
file_group.add_argument('--train_filename', type=str, help="file name of training data")
file_group.add_argument('--valid_filename', type=str, help="file name of validation data")
file_group.add_argument('--test_filename', type=str, help="file name of test data")



if __name__=='__main__':
    config = parser.parse_args()
    arg_parser = Parser()
    args = arg_parser(parser, config)
    
    seed = config.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    
    
    # 허깅페이스 데이터셋으로 변환 ------------------------
    from datasets import load_dataset, load_metric, Audio
    
    train_file = os.path.join(DATA_PATH, config.train_filename)
    test_file  = os.path.join(DATA_PATH, config.test_filename)
    
    train_dataset = load_dataset("csv", data_files={"train": train_file}, delimiter=",")["train"]
    # train = load_dataset("csv", data_files={"train": '/home/selinawisco/selina_main/asr/01_asr_train_sample.csv'}, delimiter=",")["train"]
    test = load_dataset("csv", data_files={"test": test_file}, delimiter=",")["test"]
    #remove unnecessary columns
    train_dataset = train_dataset.remove_columns(['disease_type', 'age', 'gender','subgroup', 'id', 'textgrid_text', 'target_text','asr_text', 'new_label' ])
    test = test.remove_columns(['disease_type', 'age', 'gender','subgroup', 'id', 'textgrid_text', 'target_text','asr_text', 'new_label'])

    # train = train.rename_column("speech_file", "audio")
    # test = test.rename_column("speech_file", "audio")

    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16_000))
    test = test.cast_column("audio", Audio(sampling_rate=16_000))


    # 모델 정의 ----------------------------------------
    from transformers import Wav2Vec2CTCTokenizer
    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token='[UNK]', pad_token = "[PAD]", word_delimeter_token="|")

    from transformers import Wav2Vec2FeatureExtractor
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

    from transformers import Wav2Vec2Processor
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    
    # # 데이터셋 프리 프로세싱 ------------------------------

    def prepare_dataset(batch):
        
        audio = batch['audio']
        batch['input_values'] = processor(audio['array'], sampling_rate=audio['sampling_rate']).input_values[0]
        
        if config.loss_feature != None: # if none -> baseline

            loss_feature = batch[config.loss_feature]  # List of cer_word entries corresponding to each audio
            orig_input = batch['input_values']
            new_input = np.append(orig_input, loss_feature).tolist() # append it at the end
            batch['input_values'] = new_input

        # Extract labels (transcripts) as usual
        with processor.as_target_processor():
            batch['labels'] = processor(batch[config.target]).input_ids

        return batch

    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names,num_proc=1)
    test_dataset = test.map(prepare_dataset, remove_columns=test.column_names,num_proc=1)
    
    
    # DATA COLLATOR 패딩 함수 정의 ------------------------
    from dataclasses import dataclass, field
    from typing import Any, Dict, List, Optional, Union

    @dataclass
    class DataCollatorCTCWithPadding:
        """
        Data collator that will dynamically pad the inputs received.
        Args:
            processor (:class:`~transformers.Wav2Vec2Processor`)
                The processor used for proccessing the data.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
                among:
                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
            max_length_labels (:obj:`int`, `optional`):
                Maximum length of the ``labels`` returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
                7.5 (Volta).
        """
        
        processor: Wav2Vec2Processor
        padding: Union[bool,str]= True
        max_length: Optional[int] = None
        max_length_labels: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        pad_to_multiple_of_labels: Optional[int] = None
        
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            
                
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            
            if config.loss_feature != None:
                loss_feature_batch = torch.tensor([feature['input_values'][-1] for feature in input_features], dtype=torch.float32)
            
                # remove cer_word from input values
                for feature in features:
                    feature['input_values'] = feature['input_values'][:-1]

            batch = self.processor.pad(
                input_features,
                padding = self.padding,
                max_length = self.max_length,
                pad_to_multiple_of = self.pad_to_multiple_of,
                return_tensors="pt",
            )
            
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding = self.padding,
                    max_length = self.max_length_labels,
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )
            
            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels
            
            if config.loss_feature != None:
                batch[config.loss_feature] = loss_feature_batch
            
            return batch
        
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    # 매트릭 정의--------------------------------------------
    import numpy as np
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        for i in range(3):
            print(f'Prediction: {pred_str[i]}')
            print(f"Label: {label_str[i]}")
        
        # wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = jiwer.cer(pred_str, label_str)
        return {"batch_cer": cer}
    
    
    # 모델 생성 ---------------------------------------
    from transformers import Wav2Vec2ForCTC

    for seed in range(42, 42+config.num_runs):  # Modify this range to run the script 5 times with incrementing seeds
        print(f"Running training with seed {seed}")

        # Set the new seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
            
        model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-xls-r-300m", 
            attention_dropout=0.0,
            hidden_dropout=0.0,
            feat_proj_dropout=0.0,
            mask_time_prob=0.05,
            layerdrop=0.0,
            ctc_loss_reduction="mean", 
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
        )
        model.to(device)


        # Wrap the model with DataParallel for parallel processing when multi GPU
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        
        if config.dropout != None :
            dropout_input = config.dropout.split(":")
            dropout_rate = float(dropout_input[0])
            dropout_layers = list(map(int, dropout_input[1].split(' '))) #  ex. --dropout = "0.1:6 7 8" -> dropout_rate=0.1, layers_to_apply = [6,7,8]
            
        model.freeze_feature_encoder()
        model.gradient_checkpointing_enable()
        
        if not config.eval_mode:
            train_asr(model, data_collator, processor, tokenizer,feature_extractor, train_dataset, test_dataset, compute_metrics, config)