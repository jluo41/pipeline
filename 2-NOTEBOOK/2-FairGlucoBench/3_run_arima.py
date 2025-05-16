import sys
import os
import logging
import pandas as pd
import datasets
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# Calculate AUC (Area Under the ROC Curve)
# For multi-class, we use one-vs-rest approach
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error


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



HFDataName = 'FairGlucoBench-Bf24h-Af8h-Split'
path = os.path.join(SPACE['DATA_HFDATA'], HFDataName)
split_to_dataset = datasets.load_from_disk(path)
remove_unused_columns = True # if using the processed dataset, set to True. 
print(split_to_dataset)
Name_to_Data = {i: {'ds_tfm': split_to_dataset[i]} for i in split_to_dataset}
# exit()



def arima_forecast_with_rmse(example):
    series = np.array(example['input_ids'])
    labels = np.array(example['labels'])
    forecast_horizon = len(labels)

    # Fit ARIMA
    model = auto_arima(
        series,
        start_p=0, start_q=0,
        max_p=5, max_q=5,
        d=None,
        seasonal=False,
        stepwise=True,
        error_action='ignore',
        suppress_warnings=True,
        trace=False
    )

    # Forecast
    forecast = model.predict(n_periods=forecast_horizon)

    # RMSE computation
    rmse_results = {}
    for n in [6, 12, 24, 72]:
        if len(labels) >= n:
            rmse = np.sqrt(mean_squared_error(labels[:n], forecast[:n]))
            rmse_results[f'rmse@{n}'] = rmse
        else:
            rmse_results[f'rmse@{n}'] = None  # Not enough labels

    results = {}
    # results['rmse_results'] = rmse_results
    results['forecast'] = forecast.tolist()
    # results['labels'] = labels.tolist()
    # results['series'] = series.tolist()
    results.update(rmse_results)
    return results


if __name__ == '__main__':
    # split_names = ['test-od', 'test-id']
    split_names = ['test-id']
    for split_name in split_names:
        dataset = Name_to_Data[split_name]['ds_tfm']
        # dataset = dataset.shuffle(seed=42).select(range(100))
        dataset = dataset.map(arima_forecast_with_rmse, num_proc=16)
        # dataset
        columns = ['stratum', 'split_timebin', 'rmse@6', 'rmse@12', 'rmse@24', 'rmse@72']
        df = dataset.select_columns(columns)
        # df[columns].describe()
        path = os.path.join(SPACE['MODEL_ROOT'], 'ARIMA', HFDataName, f'{split_name}_arima_rmse.csv')
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        df.to_csv(path, index=False)
        logger.info(f"ARIMA results saved to {path}")


# python 2-NOTEBOOK/2-FairGlucoBench/3_run_arima.py
