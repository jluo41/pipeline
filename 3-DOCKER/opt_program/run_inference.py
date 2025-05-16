import os
import sys
import shutil
import logging
import datasets
import pandas as pd 
from datasets import disable_caching
from pprint import pprint 
from datetime import datetime 

def process_inference_SPACE(SPACE, MODEL_VERSION):

    assert 'MODEL_ROOT' in SPACE, "Invalid SPACE: missing MODEL_ROOT"   
    
    # pipeline from ModelVersion/pipeline
    SPACE['CODE_FN'] = os.path.join(SPACE['MODEL_ROOT'], MODEL_VERSION, 'pipeline')
    assert os.path.exists(SPACE['CODE_FN']), f"Invalid CODE_FN: {SPACE['CODE_FN']}"
    # external from ModelVersion/external
    SPACE['DATA_EXTERNAL'] = os.path.join(SPACE['MODEL_ROOT'], MODEL_VERSION, 'external')
    assert os.path.exists(SPACE['DATA_EXTERNAL']), f"Invalid DATA_EXTERNAL: {SPACE['DATA_EXTERNAL']}"

    SPACE['DATA_RAW'] = os.path.join(SPACE['MODEL_ROOT'], MODEL_VERSION)
    assert os.path.exists(SPACE['DATA_RAW']), f"Invalid DATA_EXTERNAL: {SPACE['DATA_RAW']}"

    SPACE['DATA_INFERENCE'] = os.path.join(SPACE['MODEL_ROOT'], MODEL_VERSION, 'Inference')
    assert os.path.exists(SPACE['DATA_INFERENCE']), f"Invalid DATA_EXTERNAL: {SPACE['DATA_INFERENCE']}"

    SPACE['MODEL_VERSION'] = MODEL_VERSION
    return SPACE


MODEL_VERSION = 'vTestJardianceRxEgm'    # 'vTest'
MODEL_ROOT = '../../../_Model' # '../opt_ml/model'
INF_CohortName = '20240410_Inference'
POST_PROCESS_NAME = "EngagementPredToLabel"
LoggerLevel = "INFO"

#############################
# MODEL_VERSION  = os.environ.get('MODEL_VERSION', MODEL_VERSION)
# INF_CohortName = os.environ.get('INF_COHORT_NAME', '20241013_InferencePttSampleV0')
# MODEL_ROOT     = os.environ.get('MODEL_ROOT', MODEL_ROOT)
SPACE          = {'MODEL_ROOT': MODEL_ROOT}  
SPACE = process_inference_SPACE(SPACE, MODEL_VERSION)
#############################

if SPACE['CODE_FN'] not in sys.path:
    sys.path.append(SPACE['CODE_FN'])
    sys.path = list(set(sys.path))


from config.config_record.Cohort import CohortName_to_OneCohortArgs
from recfldtkn.record_base.cohort import CohortFn, Cohort
from recfldtkn.case_base.caseutils import get_ROCOGammePhiInfo_from_CFList
from recfldtkn.aidata_base.aidata_base import get_CFList_from_AIDataBase

from recfldtkn.aidata_base.aidata_base import AIData_Base 
from recfldtkn.record_base.record_base import Record_Base
from recfldtkn.case_base.case_base import Case_Base
from recfldtkn.model_base.model_base import Model_Base
from recfldtkn.base import fill_missing_keys

from nn import Name_to_ModelInstanceClass

from inference.utils_inference import (
    load_AIData_Model_InfoSettings,
    load_Inference_Entry_Example,
    pipeline_inference_for_modelbase,
    Record_Proc_Config,
    Case_Proc_Config,
    AIDataArgs_columns
)

from inference.post_process import (
    POST_PROCESS_NAME_TO_FUNCTION
)

if __name__ == "__main__":


    Context = load_AIData_Model_InfoSettings( INF_CohortName, 
                                            Name_to_ModelInstanceClass, 
                                            CohortName_to_OneCohortArgs,

                                            Record_Proc_Config,
                                            Case_Proc_Config,
                                            AIDataArgs_columns,

                                            get_CFList_from_AIDataBase, 
                                            get_ROCOGammePhiInfo_from_CFList, 

                                            AIData_Base,
                                            Model_Base, 
                                            
                                            SPACE)

    model_base = Context['model_base']
    aidata_base = Context['aidata_base']
    InfoSettings = Context['InfoSettings']

    # --------- prepare Inference Entry ---------
    ################
    Inference_Entry_Example = load_Inference_Entry_Example(INF_CohortName, 
                                                            CohortName_to_OneCohortArgs,
                                                            Cohort,
                                                            CohortFn,
                                                            SPACE)
    ########################
    
    # Inference_Entry = 
    
    Inference_Entry = {}
    Inference_Entry['inference_form'] = Inference_Entry_Example['inference_form']
    Inference_Entry['template_form']  = Inference_Entry_Example['template_form']
    if 'TriggerName_to_dfCaseTrigger' in Inference_Entry_Example:
        Inference_Entry['TriggerName_to_dfCaseTrigger'] = Inference_Entry_Example['TriggerName_to_dfCaseTrigger']
            
    pprint(Inference_Entry, sort_dicts=False, compact=True)
    # --------- pipeline_inference_for_modelbase ---------
    inference_results = pipeline_inference_for_modelbase(
            Inference_Entry = Inference_Entry,
            Record_Base = Record_Base, 
            Case_Base = Case_Base,
            aidata_base = aidata_base, 
            model_base = model_base,
            InfoSettings = InfoSettings, 
            SPACE = SPACE)
    # ----------------------------------------------------
    du1 = inference_results['du1']
    du2 = inference_results['du2']
    du3 = inference_results['du3']
    du4 = inference_results['du4']
    total_time = inference_results['total_time']
    
    ModelCheckpointName_to_InferenceInfo = inference_results['ModelCheckpointName_to_InferenceInfo']
    
    ModelCheckpointName_to_InferenceInfo = {
        k: {k1: [round(float(i), 4) for i in list(v1)] for k1, v1 in v.items()} for k, v in ModelCheckpointName_to_InferenceInfo.items()
    }
    
    PostFn = POST_PROCESS_NAME_TO_FUNCTION[POST_PROCESS_NAME]
    
    print('record_base:', du1)
    print('case_base:', du2)
    print('aidata_base and model_base update:', du3)
    print('model_infernece:', du4)
    print('total_time:', total_time)
    
    
    results = PostFn(ModelCheckpointName_to_InferenceInfo)
    pprint(results, sort_dicts=False, compact=True)
