# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function
import io
import flask
import logging
import traceback
from flask import Flask, request, jsonify, Response
# from postprocess import convert_modeloutput_to_optimaltiminglabels, pick_up_best_RxEngagement
from context import (
    model_base, 
    aidata_base,
    InfoSettings,
    Inference_Entry_Example,
    POST_PROCESS_NAME_TO_FUNCTION,
    POST_PROCESS_NAME,
    LoggerLevel,
    fill_missing_keys,
    pipeline_inference_for_modelbase,
    Record_Base,
    Case_Base,
    SPACE
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s:%(asctime)s:(%(filename)s@%(lineno)d %(name)s)]: %(message)s')
logger = logging.getLogger(__name__)
# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

# The flask app for serving predictions
app = Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here
    health = True
    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    try:
        #####################
        inference_form = request.get_json(force=True)
        # TODO: add the data_form, ds_case_form, etc. 
        template_form = Inference_Entry_Example['template_form']
        # inference_form = update_medication_strength(inference_form)  
        inference_form = fill_missing_keys(inference_form, template_form)

        Inference_Entry = {}
        Inference_Entry['inference_form'] = inference_form
        Inference_Entry['template_form'] = template_form

        if 'TriggerName_to_dfCaseTrigger' in Inference_Entry_Example:
            Inference_Entry['TriggerName_to_dfCaseTrigger'] = Inference_Entry_Example['TriggerName_to_dfCaseTrigger']

        # TODO: assert TriggerName_to_dfcasetrigger
        #####################
        
        # --------- pipeline_inference_for_modelbase ---------
        inference_results = pipeline_inference_for_modelbase(
            Inference_Entry = Inference_Entry_Example,
            Record_Base = Record_Base, 
            Case_Base = Case_Base,
            aidata_base = aidata_base, 
            model_base = model_base,
            InfoSettings = InfoSettings, 
            SPACE = SPACE
        )
        ModelCheckpointName_to_InferenceInfo = inference_results['ModelCheckpointName_to_InferenceInfo']
        
        ModelCheckpointName_to_InferenceInfo = {
            k: {k1: [round(float(i), 4) for i in list(v1)] for k1, v1 in v.items()} for k, v in ModelCheckpointName_to_InferenceInfo.items()
        }
        
        PostFn = POST_PROCESS_NAME_TO_FUNCTION[POST_PROCESS_NAME]
        if LoggerLevel == 'INFO': 
            # ----------------------------------------------------
            du1 = inference_results['du1']
            du2 = inference_results['du2']
            du3 = inference_results['du3']
            du4 = inference_results['du4']
            total_time = inference_results['total_time']
        
            logger.info(ModelCheckpointName_to_InferenceInfo)
            logger.info(f'record_base: {du1}')
            logger.info(f'case_base: {du2}')
            logger.info(f'aidata_base and model_base update: {du3}')
            logger.info(f'model_infernece: {du4}')
            logger.info(f'total_time: {total_time}')
            

        results = PostFn(ModelCheckpointName_to_InferenceInfo)
        results = jsonify(results), 200
        return results
    
    except Exception as e:
        trc = traceback.format_exc()
        message = 'Exception during inference: ' + str(e) + '\n' + trc
        logger.error(f"Error during inference: {message}")
        return jsonify({
            "status": {
                "code": 500,
                "message": message,
            }
        }), 500

