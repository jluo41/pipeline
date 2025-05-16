import os
import sys
import shutil
import logging
import datasets
import pandas as pd 
from datasets import disable_caching
from pprint import pprint 
from datetime import datetime 


def process_inference_SPACE(SPACE, MODEL_ENDPOINT):

    assert 'MODEL_ROOT' in SPACE, "Invalid SPACE: missing MODEL_ROOT"   
    
    # pipeline from ModelVersion/pipeline
    SPACE['CODE_FN'] = os.path.join(SPACE['MODEL_ROOT'], MODEL_ENDPOINT, 'pipeline')
    assert os.path.exists(SPACE['CODE_FN']), f"Invalid CODE_FN: {SPACE['CODE_FN']}"
    # external from ModelVersion/external
    SPACE['DATA_EXTERNAL'] = os.path.join(SPACE['MODEL_ROOT'], MODEL_ENDPOINT, 'external')
    assert os.path.exists(SPACE['DATA_EXTERNAL']), f"Invalid DATA_EXTERNAL: {SPACE['DATA_EXTERNAL']}"

    SPACE['DATA_RAW'] = os.path.join(SPACE['MODEL_ROOT'], MODEL_ENDPOINT)
    assert os.path.exists(SPACE['DATA_RAW']), f"Invalid DATA_EXTERNAL: {SPACE['DATA_RAW']}"

    SPACE['DATA_INFERENCE'] = os.path.join(SPACE['MODEL_ROOT'], MODEL_ENDPOINT, 'inference')
    assert os.path.exists(SPACE['DATA_INFERENCE']), f"Invalid DATA_EXTERNAL: {SPACE['DATA_INFERENCE']}"

    SPACE['MODEL_ENDPOINT'] = MODEL_ENDPOINT
    return SPACE



### weight prediction

############################
# ----------- environment for Estimator.deploy() -----------
MODEL_ROOT          = '../../../_Model'           # '/opt/ml/model' in sagemaker
MODEL_ENDPOINT      = 'vTestWeight' # 'vTestCGMFull'
INF_CohortName      = '20241013_InferencePttSampleV0'
INF_OneCohortArgs   = {'CohortLabel': 9,
                       'CohortName': '20241013_InferencePttSampleV0',
                       'FolderPath': '$DATA_RAW$/inference/',
                       'SourcePath': 'patient_sample',
                       'Source2CohortName': 'InferencePttSampleV0'}
INF_CFArgs          = None 
INF_Args            = None 

PostFnName = "PostFn_NaiveForUniLabelPred" # "EngagementPredToLabel"
TrigFnName = 'TriggerFn_WeightEntry_v1211' 
MetaFnName = 'MetaFn_None'

POST_PROCESS_SCRIPT = None # 'pipeline/inference/post_process.py' # by default, use this script
LoggerLevel         = "INFO"
############################


############################
# MODEL_ROOT          = '../../../_Model'           # '/opt/ml/model' in sagemaker
# MODEL_ENDPOINT      = 'vTestCGMFull' # 'vTestWeight' # 
# INF_CohortName      = '20241013_InferencePttSampleV0'
# INF_OneCohortArgs   = {'CohortLabel': 9,
#                        'CohortName': '20241013_InferencePttSampleV0',
#                        'FolderPath': '$DATA_RAW$/inference/',
#                        'SourcePath': 'patient_sample',
#                        'Source2CohortName': 'InferencePttSampleV0'}
# INF_CFArgs          = ['cf.TargetCGM_Bf24H'] 
# INF_Args            = {'GEN_Args': {
#                             'num_first_tokens_for_gen': 289,
#                             'max_new_tokens': 24,
#                             'do_sample': False,
#                             'items_list': ['hist', 'pred', 'logit_scores']}
#                       } 
# MetaFnName = 'MetaFn_None'
# TrigFnName = 'TriggerFn_CGM5MinEntry_v1211' 
# PostFnName = "PostFn_WithCGMPred_v1210" # "EngagementPredToLabel"
# POST_PROCESS_SCRIPT = None # 'pipeline/inference/post_process.py' # by default, use this script
# LoggerLevel         = "INFO"
############################


############################# # image your are in the sagemaker container
MODEL_ROOT        = os.environ.get('MODEL_ROOT', MODEL_ROOT)
MODEL_ENDPOINT    = os.environ.get('MODEL_ENDPOINT', MODEL_ENDPOINT)
INF_CohortName    = os.environ.get('INF_COHORT_NAME', INF_CohortName)
INF_CohortArgs    = os.environ.get('INF_COHORT_ARGS', INF_OneCohortArgs)
InputCFArgs_ForInference = os.environ.get('INF_CFArgs', INF_CFArgs)
InferenceArgs     = os.environ.get('INF_Args', INF_Args)   

PostFnName = os.environ.get('PostFnName', PostFnName)
TrigFnName = os.environ.get('TrigFnName', TrigFnName)
MetaFnName = os.environ.get('MetaFnName', MetaFnName)

LoggerLevel       = os.environ.get('LOGGER_LEVEL', LoggerLevel)
#############################



SPACE = {'MODEL_ROOT': MODEL_ROOT}  
SPACE = process_inference_SPACE(SPACE, MODEL_ENDPOINT)
if SPACE['CODE_FN'] not in sys.path:
    sys.path.append(SPACE['CODE_FN'])
    sys.path = list(set(sys.path))


from recfldtkn.record_base.cohort import CohortFn, Cohort
from recfldtkn.case_base.caseutils import get_ROCOGammePhiInfo_from_CFList
from recfldtkn.aidata_base.aidata_base import AIData_Base 
from recfldtkn.record_base.record_base import Record_Base
from recfldtkn.case_base.case_base import Case_Base
from recfldtkn.model_base.model_base import Model_Base
from recfldtkn.base import fill_missing_keys


from nn import load_model_instance_from_nn

from inference.utils_inference import (
    load_AIData_Model_InfoSettings,
    load_Inference_Entry_Example,
    pipeline_inference_for_modelbase,
    Record_Proc_Config,
    Case_Proc_Config,
    OneEntryArgs_items_for_inference,
)

from inference.post_process import NAME_TO_FUNCTION

# TODO: input with a script.py and update this NAME_TO_FUNCTION # dict

if __name__ == "__main__":
    
    MetaFn = NAME_TO_FUNCTION[MetaFnName]
    TrigFn = NAME_TO_FUNCTION[TrigFnName]
    PostFn = NAME_TO_FUNCTION[PostFnName]
    
    
    # --------- meta_results ---------
    meta_results = MetaFn(SPACE)
    if meta_results is None:
        print('No meta_results')
    else:
        metadata_response = meta_results.get('metadata_response', None)
        pprint('metadata_response:', metadata_response)
    
    
    # --------- load context ---------

    ModelEndpoint_Path = os.path.join(SPACE['MODEL_ROOT'], SPACE['MODEL_ENDPOINT'])
    assert os.path.exists(ModelEndpoint_Path), f"Invalid ModelEndpoint_Path: {ModelEndpoint_Path}"

    CohortName_to_OneCohortArgs = {INF_CohortName: INF_OneCohortArgs}

    Package_Settings = {
        'INF_CohortName': INF_CohortName,
        'INF_OneCohortArgs': INF_OneCohortArgs,
        'Record_Proc_Config': Record_Proc_Config,
        'Case_Proc_Config': Case_Proc_Config,
        'OneEntryArgs_items_for_inference': OneEntryArgs_items_for_inference,
        'get_ROCOGammePhiInfo_from_CFList': get_ROCOGammePhiInfo_from_CFList,
        'load_model_instance_from_nn': load_model_instance_from_nn,
        'Model_Base': Model_Base,
        'AIData_Base': AIData_Base,
    }

    Context = load_AIData_Model_InfoSettings(
        ModelEndpoint_Path = ModelEndpoint_Path,
        InputCFArgs_ForInference = InputCFArgs_ForInference, 
        InferenceArgs = InferenceArgs, 
        SPACE = SPACE,
        **Package_Settings,
    )

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
    # inference_form = Inference_Entry_Example['inference_form']
    inference_form_name = [i for i in Inference_Entry_Example if 'inference_form' in i][0]
    inference_form = Inference_Entry_Example[inference_form_name]

    

    # get ModelSeries_external_to_call
    ModelArtifacts_external_to_call = inference_form.get('models')
    # ModelArtifacts_external_to_call

    if ModelArtifacts_external_to_call is not None:
        ModelArtifacts_to_call = [meta_results['External_to_Local_ModelArtifacts'][i] for i in ModelArtifacts_external_to_call]
    else:
        ModelArtifacts_to_call = None 
        
    # get TriggerName_to_CaseTriggerList
    ######################################
    TriggerName_to_CaseTriggerList = None
    ######################################
    TrigFn = NAME_TO_FUNCTION.get(TrigFnName)
    if TriggerName_to_CaseTriggerList is None: 
        TriggerName_to_CaseTriggerList = TrigFn(inference_form)
    TriggerName_to_dfCaseTrigger = {k: pd.DataFrame(v) for k, v in TriggerName_to_CaseTriggerList.items()}

    
    # prepare Inference Entry
    template_form = Inference_Entry_Example['template_form']
    inference_form = fill_missing_keys(inference_form, template_form)
    
    Inference_Entry = {}
    Inference_Entry['inference_form'] = inference_form
    Inference_Entry['template_form']  = Inference_Entry_Example['template_form']
    Inference_Entry['TriggerName_to_dfCaseTrigger'] = TriggerName_to_dfCaseTrigger
    Inference_Entry['ModelArtifacts_to_call'] = ModelArtifacts_to_call

         
    pprint([i for i in Inference_Entry], sort_dicts=False, compact=True)
    
    # --------- pipeline_inference_for_modelbase ---------
    inference_results = pipeline_inference_for_modelbase(
        Inference_Entry = Inference_Entry,
        Record_Base = Record_Base, 
        Case_Base = Case_Base,
        aidata_base = aidata_base, 
        model_base = model_base,
        InfoSettings = InfoSettings, 
        SPACE = SPACE
    )
    # ----------------------------------------------------
    du1 = inference_results['du1']
    du2 = inference_results['du2']
    du3 = inference_results['du3']
    du4 = inference_results['du4']
    total_time = inference_results['total_time']
    
    ModelArtifactName_to_Inference = inference_results['ModelArtifactName_to_Inference']
    
    print('record_base:', du1)
    print('case_base:', du2)
    print('aidata_base and model_base update:', du3)
    print('model_infernece:', du4)
    print('total_time:', total_time)
    
    pprint(ModelArtifactName_to_Inference, sort_dicts=False, compact=True)
    
    
    results = PostFn(ModelArtifactName_to_Inference, SPACE)
    print('PostFnName', PostFnName)
    pprint(results, sort_dicts=False, compact=True)
