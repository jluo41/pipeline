import os
import logging
import pandas as pd 
from pprint import pprint 
import sys

SPACE = {
    'DATA_RAW': f'_Data/0-Data_Raw',
    'DATA_RFT': f'_Data/1-Data_RFT',
    'DATA_CASE': f'_Data/2-Data_CASE',
    'DATA_AIDATA': f'_Data/3-Data_AIDATA',
    'DATA_EXTERNAL': f'code/external',
    'DATA_SPLIT': f'_Data/4-Data_Split',
    'CODE_FN': f'code/pipeline', 
}

sys.path.append(SPACE['CODE_FN'])
# SPACE['WORKSPACE_PATH'] = WORKSPACE_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s:%(asctime)s:(%(filename)s@%(lineno)d %(name)s)]: %(message)s')

from datasets import disable_caching
disable_caching()

from config.config_record.Cohort import CohortName_to_OneCohortArgs
from config.config_case.TagRec import TagRec_to_TagRecArgs
from config.config_case.Flt import FltName_to_FltArgs
# from config.config_case.CASE import TriggerCaseBaseName_to_TriggerCaseBaseArgs
from datasets.fingerprint import Hasher 
from recfldtkn.check import update_and_assert_CaseInfo
from recfldtkn.check import retrive_pipeline_info

from recfldtkn.record_base import Record_Base
from recfldtkn.case_base.case_base import Case_Base

import argparse



# Parse command line arguments
parser = argparse.ArgumentParser(description='Run casebase for CGM food data')
parser.add_argument('--cohort', type=str, default='WellDoc2022CGM', 
                    help='Cohort name to process')
parser.add_argument('--n_cpus', type=int, default=6, 
                    help='Number of CPUs to use')
args = parser.parse_args()

CohortName = args.cohort
n_cpus = args.n_cpus

Case_Args_Settings = {
    'TagRec_to_TagRecArgs': TagRec_to_TagRecArgs,
    'FltName_to_FltArgs': FltName_to_FltArgs,
}


###################################
Record_Proc_Config = {
    'save_data': True, 
    'load_data': True, 
    'via_method': 'ds',
    'shadow_df': False, # record.df_RecAttr (True) or ds_RecAttr (False)
}

Case_Proc_Config = {
    'max_trigger_case_num': None, 
    'use_task_cache': False, 
    'caseset_chunk_size': 200000, # 200k for CGM, 50k for others.
    'save_data': True, 
    'load_data': True, 
    'load_casecollection': False,
    'via_method': 'ds',
    'n_cpus': n_cpus, 
    'batch_size': 1000,  
    'save_memory': False, # True, # <--- turn it to off when we want to have the results. 
}
###################################



# NOTEBOOK/1-v250420-TriggerAllCGMCase-WellDoc/250312-WellDoc2025Data


# python 2-NOTEBOOK/2-FairGlucoBench/run_case_fairgluco.py --cohort WellDoc2022CGM --n_cpus 4
# python 2-NOTEBOOK/2-FairGlucoBench/run_case_fairgluco.py --cohort WellDoc2025ALS --n_cpus 1
# python 2-NOTEBOOK/2-FairGlucoBench/run_case_fairgluco.py --cohort WellDoc2025CVS --n_cpus 4
# python 2-NOTEBOOK/2-FairGlucoBench/run_case_fairgluco.py --cohort WellDoc2025LLY --n_cpus 4


CohortName_list = [
    CohortName,
    # 'WellDoc2022CGM', 
    # 'WellDoc2025ALS',
    # 'WellDoc2025CVS',
    # 'WellDoc2025LLY',
]

# --------------------------------------------------
TriggerCaseBaseArgs = {

    # --------- this three are relatively stable ----------------
    'Trigger': {
        'TriggerName': 'CGM5MinEntry', 
        'CaseFnTasks': [
            'CGMValueBf24h',
            'CGMValueAf2h',
            'CGMValueAf2to8h',
        ]
    },
    # --------------------------------
}



########################################################
dataset_init_path = os.path.join(SPACE['DATA_SPLIT'], 'WellDoc_ds_case_fairglucose_split.parquet')
dataset_init = pd.read_parquet(dataset_init_path)

dataset_init_name = 'FairGlu'
dataset_init_info = {
    'dataset_init': dataset_init,
    'dataset_init_name': dataset_init_name,
}


TriggerName = TriggerCaseBaseArgs['Trigger']['TriggerName']
TriggerCaseBaseName = f'{dataset_init_name}-{TriggerName}-' + Hasher().hash(TriggerCaseBaseArgs)
######################################################## 




if __name__ == '__main__':

    PIPELINE_INFO = retrive_pipeline_info(SPACE)
    
    
    CaseSettingInfo = update_and_assert_CaseInfo(
                                TriggerCaseBaseName,
                                TriggerCaseBaseArgs,
                                Case_Args_Settings,
                                Case_Proc_Config, 
                                PIPELINE_INFO, 
                                SPACE)

    
    HumanRecordRecfeat_Args = CaseSettingInfo['HumanRecordRecfeat_Args']
    HumanRecordRecfeat_Args['P']['CGM5Min'] = []

    # HumanRecordRecfeat_Args[('P', 'CGM5Min')] = []
    record_base = Record_Base(CohortName_list, 
                                HumanRecordRecfeat_Args,
                                CohortName_to_OneCohortArgs,
                                SPACE = SPACE, 
                                # Inference_Entry = Inference_Entry,
                                Record_Proc_Config = Record_Proc_Config,
                                )

    # CohortTriggerCaseBaseArgs = Name_to_CohortTriggerCaseBaseArgs[TriggerCaseBaseName]
    TriggerCaseBaseName_to_TriggerCaseBaseArgs = {}
    TriggerCaseBaseName_to_TriggerCaseBaseArgs[TriggerCaseBaseName] = TriggerCaseBaseArgs
    pprint(TriggerCaseBaseArgs, sort_dicts=False)


    TriggerCaseBaseName_to_CohortNameList = {
        TriggerCaseBaseName: CohortName_list,
    }

    TriggerCaseBaseName_to_dataset_init = {
        TriggerCaseBaseName: dataset_init_info,
    }

    case_base = Case_Base(
        record_base = record_base, 
        TriggerCaseBaseName_to_CohortNameList      = TriggerCaseBaseName_to_CohortNameList, 
        TriggerCaseBaseName_to_TriggerCaseBaseArgs = TriggerCaseBaseName_to_TriggerCaseBaseArgs,
        TriggerCaseBaseName_to_dataset_init        = TriggerCaseBaseName_to_dataset_init,
        Case_Proc_Config = Case_Proc_Config,
        Case_Args_Settings = Case_Args_Settings, 
    )


    try:
        dataset = case_base.create_dataset()
        AIDataName = TriggerCaseBaseName # config['AIDataName']
        local_path = os.path.join(SPACE['DATA_AIDATA'], CohortName, AIDataName)
        dataset.save_to_disk(local_path)
        print(dataset)
    except Exception as e:
        logger.error(f'Error saving dataset: {e}')
        raise e
