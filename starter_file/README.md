*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Capstone Project - Azure Machine Learning Engineer

This project is the capstone project for "Machine Learning Engineer for Microsoft Azure" Udacity's Nanodegree. 
In this project we will choose a public external dataset. This dataset will be used for train a model using 
1) an Automated ML 
2) Hyperdrive.

## Project Set Up and Installation
1) Create dataset with browser csv dataset file
2) Run automl.ipynb steps by steps to run config, train models and deploy best model
3) Upload envirement and train, score script for hyperparameter_tuning.ipynb
   1) myenv.yml
   2) train_validate.py
   3) score.py
   4) Run hyperparameter_tuning steps by steps

## Dataset
Dataset [Student's Scores](https://www.kaggle.com/datasets/samarsaeedkhan/scores) from kaggle

![Dataset details](./capture/dataset.PNG)

### Overview
This dataset contains the marks of students in percentage and the subjects are ML, Excel, Python, Tableau and SQL.

### Task
I am using this data in order to predict the Student Placed: whether the student is placed with these total marks or not, from the scores of the subjects 
1. Python: the percentage of the python's subject score (float: 0-1).
2. Sql: the percentage of the SQL subject score. (float: 0-1).
3. ML: the percentage of the ML subject score (float: 0-1).
4. Tableau : the percentage of the Tableau  subject score (float: 0-1).
5. Excel: the percentage of the Excel subject score (float: 0-1).
6. Student Placed: student is placed with these total marks or not (boolean: yes-no).

### Access
Create and access datasheet with data provided by the website:
URL: [https://storage.googleapis.com/kagglesdsdata/datasets/2346686/3954047/scores.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220809%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220809T152449Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=376ed0e9e7c679ea63fde67eb10bbeb8f12d462521a4e330ff5a3e8eaaa6700fbc4a5f24bef2cb662ae4c51f709b60ed8bcbf2dc326bfe006d69d725d63af142b4ceb7413ed6f8134c611e62da4ae6f02adef5a6f8208131dcfdac4de7cc86b6c6fdc10b88f3b30a22ffe2fcb2b33a5ba4d1e70635772990121f6c06625d18e936cd77942df59d72422b261f67bcc6a185b3c002d2e90e1174faf9d10ba066bfc515a31754344c112772754656de7fbca9f9f46d8dcba1e37342793e9b79f35c29ec3c723d720be26c1b6b1bbb6601bf22c84b063ebc1e975e1f00179cea6214cb96ac6173f5b7bf3d7009b6d8986003a7c25fa022bc1795c9e2de70879d0536
](https://storage.googleapis.com/kagglesdsdata/datasets/2346686/3954047/scores.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220809%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220809T152449Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=376ed0e9e7c679ea63fde67eb10bbeb8f12d462521a4e330ff5a3e8eaaa6700fbc4a5f24bef2cb662ae4c51f709b60ed8bcbf2dc326bfe006d69d725d63af142b4ceb7413ed6f8134c611e62da4ae6f02adef5a6f8208131dcfdac4de7cc86b6c6fdc10b88f3b30a22ffe2fcb2b33a5ba4d1e70635772990121f6c06625d18e936cd77942df59d72422b261f67bcc6a185b3c002d2e90e1174faf9d10ba066bfc515a31754344c112772754656de7fbca9f9f46d8dcba1e37342793e9b79f35c29ec3c723d720be26c1b6b1bbb6601bf22c84b063ebc1e975e1f00179cea6214cb96ac6173f5b7bf3d7009b6d8986003a7c25fa022bc1795c9e2de70879d0536)

## Automated ML
1. AutoML Setting:
```
 automl_settings = {
  "experiment_timeout_minutes": 20,
  "max_concurrent_iterations": 4,
  "primary_metric" : 'AUC_weighted'
}
```
- experiment_timeout_minutes = 20 : It defines how long, in minutes, your experiment should continue to run. In previous projects we couldn't set more than 20 minutes. In this project we could use more but it is not needed for so small training set. However, it is for sure something that you could change for better performance.

- max_concurrent_iterations = 4 : It is how many cross validations set to make when user validation data is not specified. The main set of data is split to n=4 sets and it is performed train on the one of the two and validation to the other set. So this procedure is performed two times, because we have n_cross_validations=4.

- primary_metric = 'AUC_weighted' : The metric that Automated Machine Learning will optimize for model selection. We have set the 'AUC_weighted'.

2. AutoML Config
```
automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "classification",
                             training_data=dataset,
                             label_column_name="Student Placed",   
                             path = project_folder,
                             enable_early_stopping= True,
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings
                            )
```
- compute_target = compute_target : The compute target with specific vm_size and max_nodes. The one that has been configured with name 'aml_compute' in the automl.ipynb.

- task='classification' : We have a classification task to do, we seek to predict whether or not the person will have a heart failure. With other words we are trying to predict the DEATH_EVENT.

- training_data = dataset : The data on which the algorithm will be trained.

- label_column_name='Student Placed' : The name of the column that contains the labels of the train data, i.e the target column we want to predict.

- path= project_folder : The path to the Azure ML folder of the project.

- enable_early_stopping = True : Whether to enable early termination if the score is not improving in the short term.

- featurization= 'auto' : Indicator for whether featurization step should be done automatically or not, or whether customized featurization should be used. I used auto so featurization step step should be done automatically.

- debug_log = "automl_errors.log" : The debug information are written to the automl_errors.log.

### Results

1. Best algorightm:	MaxAbsScaler
2. Run Id:	AutoML_a7080ef9-3a1a-479a-9707-848e87477dcb_50'
3. Accuracy:	0.596


#### Result Capture
![RunDetails](./capture/RunDetails.PNG)
![Bestrun](./capture/BestModel.PNG)

## Improve:
I think i can improvements to have have best result: 
1) Set bigger ```n_cross_validation``` to 6 or 7. 
2) Increase ``` "experiment_timeout_minutes": 20 ``` to 40mins to 1 hour. 
3) Disable the early stopping policy and thus set ``` enable_early_stopping = False```.

## Hyperparameter Tuning
Tha algorithm used for the training is Logistic Regression. The two hyperparameters of the Logistic Regression are tuned with the hyperdrive to find the model with the best accuracy on the test set. The two hyperparameters are the following:

1. ParameterSampling:
I chose the RandomParameterSampling, the hyperparameters are randomly selected from the search space. The search space for the two hyperaparameters is the following:
```
param_sampling = RandomParameterSampling(
    {
        '--C' : choice(0.001,0.01,0.1,1,10,20,50,100,200,500,1000),
        '--max_iter': choice(50,100,200,300)
    }
)
```

2. BanditPolicy:
```
early_termination_policy = BanditPolicy(slack_factor = 0.1, evaluation_interval = 2, delay_evaluation = 5)
```
3. HyperDriveConfig:
```
hyperdrive_run_config = HyperDriveConfig(
    estimator=estimator,
    hyperparameter_sampling=param_sampling,
    policy=early_termination_policy,
    primary_metric_name="Accuracy",
    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
    max_total_runs=16,
    max_concurrent_runs=8,
)
```
   

### Results
1. runId: HD_b3dd28f6-e77a-4f79-9648-66296c405f66_6
2. Accuracy: 0.5666666666666667
3. Regularization: 1000.0
4. Max Iterations: 200

#### Result Capture
![RunDetails](./capture/HP_RunDetails.PNG)
![Bestrun](./capture/HP_BestModel.PNG)

## Model Deployment
1. inference_config:
```
inference_config = InferenceConfig(runtime="python", entry_script="score.py", conda_file="myenv.yml")
```
2. aciconfig
```
aciconfig = AciWebservice.deploy_configuration(cpu_cores = 2, 
                                               memory_gb = 4, 
                                               tags = {'name': "Udacity", 'Type': "Hyperdrive", "Version":"1"}, 
                                               description = 'sample service for Capstone Project Hyperdrive Classifier for Online News popularity')
```

## Screen Recording
- A working model  
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

1. [Automated ML](https://youtu.be/J_kjNjIY9-w)
2. [Hyperparameter Tuning](https://youtu.be/lZoYIvS8jPI)

## Standout Suggestions
- I have to try get best run with Hyperparameter Tuning but cannot get file output.pkl
- Hyperparameter Tuning with deploy best model always get failed.
