import sys
import os
import logging
import pandas as pd
import datasets
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    RobertaConfig,
    RobertaForSequenceClassification,
    PreTrainedTokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from pprint import pprint
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
# Calculate AUC (Area Under the ROC Curve)
# For multi-class, we use one-vs-rest approach
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

# ---------------------- Workspace Setup ----------------------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s:%(asctime)s:(%(filename)s@%(lineno)d %(name)s)]: %(message)s')
logger = logging.getLogger(__name__)

SPACE = {
    'DATA_RAW': '_Data/0-Data_Raw',
    'DATA_RFT': '_Data/1-Data_RFT',
    'DATA_CASE': '_Data/2-Data_CASE',
    'DATA_AIDATA': '_Data/3-Data_AIDATA',
    'DATA_EXTERNAL': 'code/external',
    'CODE_FN': 'code/pipeline',
    'MODEL_ROOT': '_Model',
}
assert os.path.exists(SPACE['CODE_FN']), f"{SPACE['CODE_FN']} not found"
sys.path.append(SPACE['CODE_FN'])

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---------------------- Data Preparation ----------------------
from recfldtkn.aidata_base.entry import EntryAIData_Builder

OneAIDataName = 'DietEventBench'
CF_DataName = 'DietEvent-CGM5MinEntry-1ea9d787eef20fb7'
CohortName_list = ['WellDoc2022CGM', 'WellDoc2025ALS', 'WellDoc2025CVS', 'WellDoc2025LLY']
CF_DataName_list = [f'{i}/{CF_DataName}' for i in CohortName_list]

OneEntryArgs = {
    'Split_Part': {
        'SplitMethod': 'SplitFromColumns',
        'Split_to_Selection': {
            'train': {
                'Rules': [
                    ['split_timebin', 'in', ('train-early', 'valid-early')],
                    ['MEDInfoBf24h-DietRecNum', '>', 0],
                    ['MEDInfoBf24h-DietLastToNow', '>=', 120],
                    ['MEDInfoBf24h-DietLastToNow', '<=', 420],
                    ['ObsDT_Minute', '==', 0],
                ],
                'Op': 'and'
            },
            'valid': {
                'Rules': [
                    ['split_timebin', 'in', ('train-middle', 'valid-middle')],
                    ['MEDInfoBf24h-DietRecNum', '>', 0],
                    ['MEDInfoBf24h-DietLastToNow', '>=', 120],
                    ['MEDInfoBf24h-DietLastToNow', '<=', 420],
                    ['ObsDT_Minute', '==', 0],
                ],
                'Op': 'and'
            },
            'test-id': {
                'Rules': [
                    ['split_timebin', 'in', ('train-late', 'valid-late')],
                    ['MEDInfoBf24h-DietRecNum', '>', 0],
                    ['MEDInfoBf24h-DietLastToNow', '>=', 120],
                    ['MEDInfoBf24h-DietLastToNow', '<=', 420],
                    ['ObsDT_Minute', '==', 0],
                ],
                'Op': 'and'
            },
            'test-od': {
                'Rules': [
                    ['split_timebin', 'in', ('test-early', 'test-middle', 'test-late')],
                    ['MEDInfoBf24h-DietRecNum', '>', 0],
                    ['MEDInfoBf24h-DietLastToNow', '>=', 120],
                    ['MEDInfoBf24h-DietLastToNow', '<=', 420],
                    ['ObsDT_Minute', '==', 0],
                ],
                'Op': 'and'
            }
        }
    },
    'Input_Part': {
        'EntryInputMethod': '1TknInStep',
        'CF_list': [
            'CGMValueBf24h',
            # 'CGMValueAf2h',
        ],
        'BeforePeriods': ['Bf24h'],
        # 'AfterPeriods': ['Af2h'],
        'TimeIndex': True, 
        'InferenceMode': False, # True, # True, # False, # True, 
        'TargetField': 'CGMValue',
        'TargetRange': [40, 400],
    }, 
    'Output_Part': {
        'EntryOutputMethod': 'UniLabelRules',
        'CF_list': ['MEDInfoBf24h'],
        'label_rule': {
            1: ('MEDInfoBf24h-DietLastToNow', 'in', [120, 180]),
            0: ('MEDInfoBf24h-DietLastToNow', 'in', [180, 420]),
            -100: 'others'
        },
        'assertion': [('MEDInfoBf24h-DietLastToNow', 'in', [120, 420])],
        'set_transform': True,
        'num_proc': 4,
    },
}

entry = EntryAIData_Builder(OneEntryArgs=OneEntryArgs, SPACE=SPACE)
dataset = entry.merge_one_cf_dataset(CF_DataName_list)
data_config = dataset.info.config_name 
split_to_dataset = entry.split_cf_dataset(dataset)



# ---------------------- CF Vocab Setup ----------------------
CFName = 'HM5MinStep'
interval_delta = pd.Timedelta(minutes=5)
idx2tkn = [pd.Timestamp('2022-01-01 00:00:00') + interval_delta * i for i in range(24 * 12)]
idx2tkn = [f'{i.hour:02d}:{i.minute:02d}' for i in idx2tkn]
tkn2idx = {tkn: idx for idx, tkn in enumerate(idx2tkn)}
CF_to_CFvocab = data_config['CF_to_CFvocab']
CF_to_CFvocab[CFName] = {'idx2tkn': idx2tkn, 'tkn2idx': tkn2idx}

CFName = 'CGMValue'
idx2tkn = ["PAD", "UNKNOWN", "MASK"] + [f'Other_{i}' for i in range(0, 7)] + [str(i) for i in range(10, 401)]
tkn2idx = {tkn: idx for idx, tkn in enumerate(idx2tkn)}
CF_to_CFvocab[CFName] = {'idx2tkn': idx2tkn, 'tkn2idx': tkn2idx}


entry.CF_to_CFvocab = CF_to_CFvocab
Name_to_Data = entry.setup_EntryFn_to_NameToData(split_to_dataset)

# ---------------------- Tokenizer Setup ----------------------
idx2tkn = CF_to_CFvocab['CGMValue']['idx2tkn']
vocab_dict = {token: idx for idx, token in enumerate(idx2tkn)}

wordlevel = WordLevel(vocab=vocab_dict, unk_token="UNKNOWN")
tokenizer_backend = Tokenizer(wordlevel)
tokenizer_backend.pre_tokenizer = Whitespace()

tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_backend,
    unk_token="UNKNOWN",
    pad_token="PAD",
    mask_token="MASK"
)

# ---------------------- Model Config ----------------------
num_labels = len([k for k in OneEntryArgs['Output_Part']['label_rule'] if k != -100])

model_name = 'roberta-no-hm-large'
config = RobertaConfig(
    vocab_size=len(tokenizer),
    num_labels=num_labels,
)

# model_name = 'roberta-no-hm-small'
# config = RobertaConfig(
#     vocab_size=len(tokenizer),
#     # hm_vocab_size=len(hm_idx2tkn),
#     num_labels=num_labels,

#     hidden_size=60,           # half of 768
#     intermediate_size=60 * 4,    # 4× hidden_size
#     num_attention_heads=3,     # half of 12
#     num_hidden_layers=3,       # half of 12

#     # you can also tweak dropout if you like
#     hidden_dropout_prob=0.1,
#     attention_probs_dropout_prob=0.1,
# )

model = RobertaForSequenceClassification(config=config)

# ---------------------- Metrics Function ----------------------
def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    
    # Get the number of unique classes
    n_classes = logits.shape[1]
    
    # Binarize the labels for AUC calculation
    if n_classes > 2:
        # Multi-class case
        classes = np.unique(labels)
        labels_bin = label_binarize(labels, classes=classes)
        
        # Calculate AUC for each class and average
        auc = roc_auc_score(labels_bin, logits, multi_class='ovr', average='weighted')
    else:
        # Binary case
        auc = roc_auc_score(labels, logits[:, 1])
    
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}

# ---------------------- Training Arguments ----------------------
training_args = TrainingArguments(
    output_dir=os.path.join(SPACE['MODEL_ROOT'], model_name),


    do_train = True, 
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,
    
    
    logging_steps=1,


    # optimizer & scheduler
    learning_rate=2e-5,              # base LR
    weight_decay=0.01,               # L2 regularization
    warmup_steps=200,                # manually warm up for 500 steps
    # or instead of warmup_steps, you can use:
    # warmup_ratio=0.1,             # warm up for first 10% of steps
    max_grad_norm=1.0,



    load_best_model_at_end=True,
    metric_for_best_model="valid_f1",
    greater_is_better=True,

    do_eval = True, 
    eval_steps = 100, 
    eval_strategy = 'steps', 
    report_to="wandb",  # <<--- wandb integration


    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,


    # ------- do not change these -------
    remove_unused_columns=False, # <--- must be False.
    dataloader_drop_last=True,
)

# ---------------------- Trainer Setup and Training ----------------------
data_collator = DataCollatorWithPadding(tokenizer)


ds_tfm_train  = Name_to_Data['train']['ds_tfm']
ds_tfm_valid  = Name_to_Data['valid']['ds_tfm']
ds_tfm_testid = Name_to_Data['test-id']['ds_tfm']
ds_tfm_testod = Name_to_Data['test-od']['ds_tfm']


eval_dict = {
    'valid': ds_tfm_valid,
    'test-id': ds_tfm_testid,
    'test-od': ds_tfm_testod,
}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_tfm_train,
    eval_dataset=eval_dict,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate(eval_dataset=ds_tfm_testid, metric_key_prefix="testid")
trainer.evaluate(eval_dataset=ds_tfm_testod, metric_key_prefix="testod")



# ---------------------- Run Script ----------------------

'''
export WANDB_PROJECT=DietEvent-RoBERTa
python 2-NOTEBOOK/4-DietEventBench/2_run_roberta_diet_event_detection.py
'''