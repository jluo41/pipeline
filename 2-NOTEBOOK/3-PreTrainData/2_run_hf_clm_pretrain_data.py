import sys
import os
import logging
import pandas as pd
import datasets
from pprint import pprint
from datasets import DatasetInfo
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s:%(asctime)s:(%(filename)s@%(lineno)d %(name)s)]: %(message)s')

SPACE = {
    'DATA_RAW': f'_Data/0-Data_Raw',
    'DATA_RFT': f'_Data/1-Data_RFT',
    'DATA_CASE': f'_Data/2-Data_CASE',
    'DATA_AIDATA': f'_Data/3-Data_AIDATA',
    'DATA_EXTERNAL': f'code/external',
    'DATA_HFDATA': f'_Data/5-Data_HFData',
    'CODE_FN': f'code/pipeline',
    'MODEL_ROOT': f'./_Model',
}
assert os.path.exists(SPACE['CODE_FN']), f'{SPACE["CODE_FN"]} not found'
print(SPACE['CODE_FN'])
sys.path.append(SPACE['CODE_FN'])

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

from recfldtkn.aidata_base.entry import EntryAIData_Builder

OneAIDataName = 'PreTrainBench'
Task = 'CausalLM-Split'
Version = 'v0515'
DataName = f'{OneAIDataName}-{Task}-{Version}'

CF_DataName = 'PreTrainBench-CGM5MinEntry-1ea9d787eef20fb7'
CohortName_list = ['WellDoc2022CGM', 'WellDoc2025ALS', 'WellDoc2025CVS', 'WellDoc2025LLY']
CF_DataName_list = [f'{i}/{CF_DataName}' for i in CohortName_list]
entry = EntryAIData_Builder(SPACE=SPACE)
dataset = entry.merge_one_cf_dataset(CF_DataName_list)

data_config = dataset.info.config_name 
CF_to_CFvocab = data_config['CF_to_CFvocab']

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

print([i for i in CF_to_CFvocab])


OneEntryArgs = {
    # ----------------- Input Part -----------------
    'Split_Part': {
        'SplitMethod': 'SplitFromColumns',
        'Split_to_Selection': {
            'train': {
                'Rules': [
                    ['split_timebin', 'in', ('train-early',)],
                ],
                'Op': 'and'
            },
            'valid': {
                'Rules': [
                    ['split_timebin', 'in', ('train-middle',)],
                ],
                'Op': 'and'
            },
            'test-id': {
                'Rules': [
                    ['split_timebin', 'in', ('train-late', 'valid-late')],
                ],
                'Op': 'and'
            },
            'test-od': {
                'Rules': [
                    ['split_timebin', 'in', ('test-early', 'test-middle', 'test-late')],
                ],
                'Op': 'and'
            }
        }
    },

    'Input_Part': {
        'EntryInputMethod': '1TknInStepWt5MinHM',
        'CF_list': [
            'CGMValueBf24h',
            'CGMValueAf2h',
            'CGMValueAf2to8h',
        ],
        'BeforePeriods': ['Bf24h'],
        'AfterPeriods': ['Af2h', 'Af2to8h'],
        'TimeIndex': True, 
        'InferenceMode': False, # True, # True, # False, # True, 
        'TargetField': 'CGMValue',
        'TargetRange': [40, 400],
        # 'HM': None, 
        'HM': {'start': -24, 'unit': 'h', 'interval': '5m'},
    }, 
    'Output_Part': {
        'EntryOutputMethod': 'CausalLM',
        'selected_columns': ['input_ids', 'hm_ids', 'labels', 'split_timebin', 'stratum'],
        'set_transform': False,
        'num_proc': 24, 
    },
}


if __name__ == '__main__':
    entry = EntryAIData_Builder(OneEntryArgs=OneEntryArgs, SPACE=SPACE)
    entry.CF_to_CFvocab = CF_to_CFvocab

    ds = dataset#.select(range(10))
    Data = {'ds_case': ds}
    Data = entry.setup_EntryFn_to_Data(Data, CF_to_CFvocab)


    path = os.path.join(SPACE['DATA_HFDATA'], f'{DataName}')
    print(path)
    ds_tfm = Data['ds_tfm']
    dataset_info = DatasetInfo.from_dict({'config_name': data_config})
    ds_tfm.info.update(dataset_info)
    
    dataset = ds_tfm
    split_to_dataset = entry.split_cf_dataset(dataset)
    split_to_dataset.ds_tfm.save_to_disk(path)
    


# ---------------------- Run Script ----------------------
'''
python 2-NOTEBOOK/3-PreTrainData/2_run_hf_clm_pretrain_data.py
'''