# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function
import io
import os
import sys
import json
import flask
import logging
import datasets
import traceback
import pandas as pd
from pprint import pprint
from datetime import datetime 
from flask import Flask, request, jsonify, Response
# from postprocess import convert_modeloutput_to_optimaltiminglabels, pick_up_best_RxEngagement


logging.basicConfig(level=logging.INFO, format='[%(levelname)s:%(asctime)s:(%(filename)s@%(lineno)d %(name)s)]: %(message)s')
logger = logging.getLogger(__name__)
# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.



def process_inference_SPACE(SPACE, MODEL_VERSION = None):

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


##########################################
# MODEL_VERSION = 'model'    # 'vTest'
MODEL_ROOT = os.getenv( "MODEL_ROOT", '/opt/ml/model')  # '../opt_ml/model'

INF_CohortName = os.getenv( "INF_CohortName", '20240410_Inference')

MODEL_VERSION = os.getenv( "MODEL_VERSION", None)
if MODEL_VERSION is None:
    MODEL_VERSION = [i for i in os.listdir(MODEL_ROOT) if 'v' == i[0]][0]
    
    
POST_PROCESS_NAME = os.getenv( "POST_PROCESS_NAME", "EngagementPredToLabel")


LoggerLevel = os.getenv( "LoggerLevel", "INFO")
##########################################

logger.info(f"MODEL_VERSION: {MODEL_VERSION}")

MODEL_PATH = os.path.join(MODEL_ROOT, MODEL_VERSION)

logger.warning(f"MODEL_ROOT: {MODEL_ROOT}")
logger.warning(f"MODEL_PATH: {MODEL_PATH}")
logger.warning(f'MODEL_ROOT: listdir: --> {os.listdir(MODEL_ROOT)}')
logger.warning(f'MODEL_PATH: listdir: --> {os.listdir(MODEL_PATH)}')

logger.warning(f"INF_CohortName: {INF_CohortName}")
logger.warning(f"MODEL_VERSION: {MODEL_VERSION}")
logger.warning(f'POST_PROCESS_NAME: {POST_PROCESS_NAME}')
logger.warning(f'LoggerLevel: {LoggerLevel}')



# MODEL_VERSION = os.listdir(MODEL_ROOT)[0]

#############################
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


