# 1. Download Data


## Method 1: Use CDHAI's small server
If you use the CDHAI's small server, you can get the data like this:

```bash

git clone git@github.com:JHU-CDHAI/WellDoc-SPACE.git

cd WellDoc-SPACE

# you should in the usergroup: usermod -aG welldocdata username
ln -s /home/jluo41/_ProjData/WellDoc/_Data ~/WellDoc-SPACE/_Data
```

## Method 2: Use Rockfish Server

```bash
ssh login01 # only in the rockfish server. 
tmux new-session -s welldoc # first time to create the session. 
# press ctrl-b, then d to detach the session. 

tmux ls # to list the sessions. 
tmux attach -t welldoc # to attach the session. 
```

```bash

# only in the rockfish server. 
srun --partition=ica100 --gpus=1 --mem=60GB --cpus-per-task=8 --time=23:00:00 --pty /bin/bash
conda activate torch
cd workspace/WellDoc-SPACE




export WANDB_PROJECT="mlm_pretrain_c123"
python NOTEBOOK/c-Train/250401-CGM-Mask/1_train_mlm_c123_bf22.py

export WANDB_PROJECT="event_pred"
python NOTEBOOK/c-Train/250402-CGM-Event/1_train_event_pred_pretrain.py

export WANDB_PROJECT="event_pred"
python NOTEBOOK/c-Train/250402-CGM-Event/1_train_event_pred_randinit.py




# task 1: train the model
export WANDB_PROJECT="lsm-Train_BothT1T2_21to22_c123"
python NOTEBOOK/c-Train/250327-CGM-LSM/1_train_lsm_botht1t2_21to22_c123.py; exit

# task 2: test the model
export WANDB_PROJECT="lsm-Train_BothT1T2_af21_all6c"
python NOTEBOOK/c-Train/250327-CGM-LSM/1_train_lsm_botht1t2_af21_all6c.py; exit


```



# Save the model back

```bash
scp -r _Data/3-Data_AIDATA/ jluo41@login.rockfish.jhu.edu:/home/jluo41/workspace/WellDoc-SPACE/_Data
scp -r _Data/3-Data_AIDATA/Train_* jluo41@login.rockfish.jhu.edu:/home/jluo41/workspace/WellDoc-SPACE/_Data/3-Data_AIDATA/


scp ModelLhmFood.zip jluo41@10.175.198.65:2024-WellDoc-learn-SPACE
```


# Obsolete 



## Method 2: Download the data from Google Drive
Download the sample data from 

```
https://drive.google.com/open?id=1_LXK0yyY1AzR9DkLsfYm4mYXsAD7qr2k&usp=drive_fs
```

And put the `_Data` folder to the root of the repo. 

It will look like
```
_Data/
    - 0-Data_Raw
    - 9-Data_External
```

# 2. Install Requirements

```
conda create -n rft python=3.11
pip install -r requirements.txt
# then you can use rft as your working environment. 
```

# 3. Do Data Preprocessing in Case Space

```bash
cd _WellDoc-2-CASE-WorkSpace

# for weight prediction.
python run_casebase_weightpred.py

# for cgm-lsm (this is optional, you can ignore it if you want to focus in weight prediction)
python run_casebase_c3.py # c3 mean cohort-3 CVSDeRx
```

# 4. Test Weight Prediction in Local WorkSpace
```bash
cd .. # back to root directory of the repo
cd _WellDoc-4-Model-WorkSpace

# you can see job_arguments.json, which is the copy of job_arguments_weightpred.json
# if you opne run_training_local, you can see that it takes job_arguments.json as the arguments.
python run_training_local.py 
```

After the training, you can see the `_Model` folder at the root the repo. 

Then you can try to make the inference. 

The inference sample comes from `_Data/0-Data_Raw/Inference/patient_sample`.

For the data input format, you should check `_Data/0-Data_Raw/Inference/patient_sample/inference_form_template.json`.

```bash
python run_inference_local.py
```

You will see the following information.
```
{'weightpred/UniLabelPred-weightpred.Af1w-WeightPred.Af1M.WeightLossPctLarge2-XGBClassifierV0.6-2024.10.14-556770446746a3d1': {'y_pred_score': array([0.00730572], dtype=float32)}}
record_base: 0:00:13.264621
case_base: 0:00:00.025472
aidata_base and model_base update: 0:00:00.000014
model_infernece: 0:00:00.005359
total_time: 0:00:13.295466
```

Now you can see `record_base: 0:00:13.264621` took a very long time. Because the sample data here is very large. In the application, we just need a smaller one. 

# 5. Test Weight Prediction in Docker Image.

Go to `_WellDoc-5-Container-WorkSpace`


```bash
cd _WellDoc-5-Container-WorkSpace
```

## Do docker image building

```bash
bash build_local.sh "model-test" "vTest"

# (if you need sudo)
# sudo bash build_local.sh "model-test" "vTest" 
```


## Do the training

```bash
bash local_test/train_local.sh "model-test"

# (if you need sudo)
# sudo bash local_test/train_local.sh "model-test"
```


## Setup the server

```bash
bash local_test/serve_local.sh "model-test"

# (if you need sudo)
# sudo bash local_test/serve_local.sh "model-test"
```


## Test the server

Find the URL for your server.  


Open the `postman`, set up the POST requests with the above URL.

Put the input_sample from `_Data/0-Data_Raw/Inference/patient_sample/inference_form_sample_36537.json` to the raw form, and then send the request. 

Currently, the `inference_form_sample_36537.json` is too large, and might not work. 

You can use the `inference_form_template.json` as an example. 

