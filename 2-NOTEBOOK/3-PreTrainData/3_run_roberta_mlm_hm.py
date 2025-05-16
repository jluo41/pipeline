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
    'DATA_HFDATA': '_Data/5-Data_HFData',
    'CODE_FN': 'code/pipeline',
    'MODEL_ROOT': '_Model',
}
assert os.path.exists(SPACE['CODE_FN']), f"{SPACE['CODE_FN']} not found"
sys.path.append(SPACE['CODE_FN'])

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---------------------- Data Preparation ----------------------
from recfldtkn.aidata_base.entry import EntryAIData_Builder
from nn.cgmencoder.configuration_cgmencoder import RobertaWithHMConfig
from nn.cgmencoder.modeling_cgmencoder import RobertaWithHMForMaskedLM



HFDataName = 'PreTrainBench-MaskedLM-Split-v0515'
path = os.path.join(SPACE['DATA_HFDATA'], HFDataName)
split_to_dataset = datasets.load_from_disk(path)
remove_unused_columns = False # if using the processed dataset, set to True. 
print(split_to_dataset)
Name_to_Data = {i: {'ds_tfm': split_to_dataset[i]} for i in split_to_dataset}

# exit()


# ---------------------- CF Vocab Setup ----------------------
CF_to_CFvocab = {} # data_config['CF_to_CFvocab']

CFName = 'HM5MinStep'
interval_delta = pd.Timedelta(minutes=5)
idx2tkn = [pd.Timestamp('2022-01-01 00:00:00') + interval_delta * i for i in range(24 * 12)]
idx2tkn = [f'{i.hour:02d}:{i.minute:02d}' for i in idx2tkn]
tkn2idx = {tkn: idx for idx, tkn in enumerate(idx2tkn)}
CF_to_CFvocab[CFName] = {'idx2tkn': idx2tkn, 'tkn2idx': tkn2idx}

CFName = 'CGMValue'
idx2tkn = ["PAD", "UNKNOWN", "MASK"] + [f'Other_{i}' for i in range(0, 7)] + [str(i) for i in range(10, 401)]
tkn2idx = {tkn: idx for idx, tkn in enumerate(idx2tkn)}
CF_to_CFvocab[CFName] = {'idx2tkn': idx2tkn, 'tkn2idx': tkn2idx}

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
hm_idx2tkn = CF_to_CFvocab['HM5MinStep']['idx2tkn']

model_name = 'roberta-cgm-mlm-with-hm'
config = RobertaWithHMConfig(
    vocab_size=len(tokenizer),
    hm_vocab_size = len(hm_idx2tkn)
)

# model_name = 'small-roberta-cgm-mlm-with-hm'
# config = RobertaWithHMConfig(
#     vocab_size=len(tokenizer),
#     hm_vocab_size=len(hm_idx2tkn),
#     # num_labels=num_labels,
#     hidden_size=60,           # half of 768
#     intermediate_size=60 * 4,    # 4× hidden_size
#     num_attention_heads=3,     # half of 12
#     num_hidden_layers=3,       # half of 12

#     # you can also tweak dropout if you like
#     hidden_dropout_prob=0.1,
#     attention_probs_dropout_prob=0.1,
# )


model = RobertaWithHMForMaskedLM(config=config)

print(model)

# ---------------------- Metrics Function ----------------------
# Step 1: Define the compute_metrics function for masked language modeling
def compute_metrics(pred):
    logits = torch.tensor(pred.predictions)
    labels = torch.tensor(pred.label_ids)

    # Only evaluate on masked tokens (where labels != -100)
    mask = labels != -100

    # Calculate loss on masked tokens only
    loss_fct = torch.nn.CrossEntropyLoss()
    masked_lm_loss = loss_fct(
        logits.view(-1, logits.size(-1))[mask.view(-1)], 
        labels.view(-1)[mask.view(-1)]
    )

    # Get predictions for masked tokens
    predictions = torch.argmax(logits, dim=-1)

    # Calculate accuracy on masked tokens only
    correct_preds = (predictions == labels) & mask
    accuracy = correct_preds.sum().float() / mask.sum().float()

    # Calculate perplexity
    perplexity = torch.exp(masked_lm_loss)

    return {
        'masked_lm_loss': masked_lm_loss.item(),
        'perplexity': perplexity.item(),
        'accuracy': accuracy.item(),
        'num_masked_tokens': mask.sum().item()
    }



# ---------------------- Training Arguments ----------------------
training_args = TrainingArguments(
    output_dir=os.path.join(SPACE['MODEL_ROOT'], model_name),

    do_train=True,
    do_eval=True,

    num_train_epochs=1,  # ← First run with 1 epoch
    per_device_train_batch_size=256,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,  # effective batch size = 64*4 = 256

    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=1000,
    max_grad_norm=1.0,

    logging_steps=1,

    # Evaluation settings
    eval_strategy="steps",
    eval_steps=200, # 1, # 200

    save_strategy="steps", # one epoch
    save_steps=0.2,
    save_total_limit=2,

    # No best model logic for now
    # load_best_model_at_end=True,
    # metric_for_best_model="perplexity",
    # greater_is_better=False,

    report_to="wandb",
    prediction_loss_only=False,
    remove_unused_columns=False,
    dataloader_drop_last=True,

    dataloader_num_workers=8,  # ← add this to use your CPUs
)


# ---------------------- Trainer Setup and Training ----------------------
# data_collator = DataCollatorWithPadding(tokenizer)

eval_set_size = 1042
random_seed = 42

ds_tfm_train  = Name_to_Data['train']['ds_tfm']
ds_tfm_valid  = Name_to_Data['valid']['ds_tfm'].shuffle(seed=random_seed).select(range(eval_set_size))
ds_tfm_testid = Name_to_Data['test-id']['ds_tfm'].shuffle(seed=random_seed).select(range(eval_set_size))
ds_tfm_testod = Name_to_Data['test-od']['ds_tfm'].shuffle(seed=random_seed).select(range(eval_set_size))

eval_dict = {
    'valid': ds_tfm_valid,
    'test-id': ds_tfm_testid,
    'test-od': ds_tfm_testod,
}

print(ds_tfm_train)
print(eval_dict)
# exit()

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
ssh login01
tmux ls
# tmux new -s welldoc
tmux attach -t welldoc


srun --partition=ica100 --gpus=1 --mem=60GB --cpus-per-task=12 --time=23:00:00 --pty /bin/bash
conda activate torch
cd workspace/WellDoc-SPACE


export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT=MLM-RoBERTa
# python 2-NOTEBOOK/3-PreTrainData/3_run_roberta_mlm_hm.py
python 2-NOTEBOOK/3-PreTrainData/3_run_roberta_mlm_hm.py; exit


# monitor
htop -u jluo41 # change to your username
watch -n 0.1 'nvidia-smi'
'''
