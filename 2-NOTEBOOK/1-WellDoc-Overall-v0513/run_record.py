import os
import logging
import pandas as pd 
from pprint import pprint 
# pd.set_option('display.max_columns', None)
# KEY = 'NOTEBOOK'
# WORKSPACE_PATH = os.getcwd().split(KEY)[0] + KEY
# # print(WORKSPACE_PATH)
# os.chdir(WORKSPACE_PATH)
import sys

SPACE = {
    'DATA_RAW': f'_Data/0-Data_Raw',
    'DATA_RFT': f'_Data/1-Data_RFT',
    'DATA_CASE': f'_Data/2-Data_CASE',
    'DATA_CFDATA': f'_Data/3-Data_CFDATA',
    'DATA_EXTERNAL': f'code/external',
    'CODE_FN': f'code/pipeline', 
}

sys.path.append(SPACE['CODE_FN'])
# SPACE['WORKSPACE_PATH'] = WORKSPACE_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s:%(asctime)s:(%(filename)s@%(lineno)d %(name)s)]: %(message)s')

from datasets import disable_caching
disable_caching()

from config.config_record.Cohort import CohortName_to_OneCohortArgs
from recfldtkn.record_base import Record_Base
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


###################################
Record_Proc_Config = {
    'save_data': True, 
    'load_data':True, 
    'via_method': 'ds',
    # 'shadow_df': True,
}
###################################



# python 2-NOTEBOOK/1-v250420-TriggerAllCGMCase-WellDoc/250312-WellDoc2025Data/run_record.py --cohort WellDoc2022CGM
# python 2-NOTEBOOK/1-v250420-TriggerAllCGMCase-WellDoc/250312-WellDoc2025Data/run_record.py --cohort WellDoc2025ALS 
# python 2-NOTEBOOK/1-v250420-TriggerAllCGMCase-WellDoc/250312-WellDoc2025Data/run_record.py --cohort WellDoc2025CVS
# python 2-NOTEBOOK/1-v250420-TriggerAllCGMCase-WellDoc/250312-WellDoc2025Data/run_record.py --cohort WellDoc2025LLY
# python 2-NOTEBOOK/1-v250420-TriggerAllCGMCase-WellDoc/250312-WellDoc2025Data/run_record.py --cohort WellDoc2025LLY_1n3
# python 2-NOTEBOOK/1-v250420-TriggerAllCGMCase-WellDoc/250312-WellDoc2025Data/run_record.py --cohort WellDoc2025LLY_2n3
# python 2-NOTEBOOK/1-v250420-TriggerAllCGMCase-WellDoc/250312-WellDoc2025Data/run_record.py --cohort WellDoc2025LLY_3n3


CohortName_list = [
    CohortName,
    # 'WellDoc2022CGM', 
    # 'WellDoc2025ALS',
    # 'WellDoc2025CVS',
    # 'WellDoc2025LLY',
    # 'WellDoc2025LLY_1n3',
    # 'WellDoc2025LLY_2n3',
    # 'WellDoc2025LLY_3n3',
]


HumanRecordRecfeat_Args = {
    # Human
    'P': {
        # --------------------- patient ---------------------
        'P': [],  # patient
        'CGM5Min': [], # CGM5Min
        'Diet5Min': [], # Diet5Min
        'Exercise5Min': [], # Exercise5Min
        'Med5Min': [], # Med5Min.
    }
}



if __name__ == '__main__':

    # HumanRecordRecfeat_Args[('P', 'CGM5Min')] = []
    record_base = Record_Base(CohortName_list, 
                                HumanRecordRecfeat_Args,
                                CohortName_to_OneCohortArgs,
                                SPACE = SPACE, 
                                # Inference_Entry = Inference_Entry,
                                Record_Proc_Config = Record_Proc_Config,
                                )