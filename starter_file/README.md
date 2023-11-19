*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Project Overview: Azure ML Engineer Capstone

This is the final prject of Machine Learning Engineer with Microsoft Azure Nanodegree. I am leveraging an external dataset about heart attack [here](https://www.kaggle.com/code/maxim9012/heart-attack-analysis-prediction-dataset). In the project I try AutoML and Hyperdrive and deploy the best performing model. With Hyperdrive I am tuning the learning rate and the number of estimators to improve the model. After deployment I test and consume the endpoint of the best model from both approaches.

## Dataset

### Overview
Heart attack data from Kaggle. [here](https://www.kaggle.com/code/maxim9012/heart-attack-analysis-prediction-dataset)
The dataset looks like this, with the output being the target variable (heart-attack): 
<img width="687" alt="Screenshot 2023-11-15 at 15 55 21" src="https://github.com/ttschuemp/nd00333-capstone/assets/46277840/26f6ca8e-7da8-45bc-ae51-173a003d7d65">

### Task
What I would like to predict is the binary variable heart-attack with 1 having a heart-attack and 0 not having a heart-attack. I will use all of the above features to train the model and make the predictions. 

### Access
I initiated the process by manually acquiring and importing the Kaggle dataset into Azure ML Studio. This involved uploading the CSV file and formally registering it as an Azure ML dataset. The registered dataset seamlessly integrates with Jupyter notebooks and other platforms for versatile usage. Our primary focus initially revolved around accessing the data through Jupyter notebook functionalities.

## Automated ML

### AutoML Settings:

- **experiment_timeout_minutes:** Specifies the maximum time allowed for the AutoML experiment to run.
- **max_concurrent_iterations:** Sets the maximum number of parallel iterations during the experiment.
- **primary_metric:** Defines the primary metric for optimization, in this case, set to 'accuracy'.

### AutoML Config:

- **compute_target:** The Azure Machine Learning compute target where the experiment will run.
- **task:** Specifies the type of task, which is 'classification' in this scenario.
- **training_data:** The dataset used for training the models.
- **label_column_name:** The name of the column containing the target variable.
- **enable_early_stopping:** Enables early stopping to terminate poorly performing runs.
- **iterations:** Specifies the maximum number of iterations or models to try.
- **featurization:** Specifies whether to perform automatic feature engineering ('auto' in this case).
- Additional settings from the `automl_settings` dictionary.

These settings and configurations collectively define the parameters for running the AutoML experiment for classification with a focus on accuracy. Adjustments can be made based on specific project requirements and goals.


### Results
The best model was a VotingEnsemble model with an accuracy of 85%.
<img width="1413" alt="Screenshot 2023-11-13 at 13 55 04" src="https://github.com/ttschuemp/nd00333-capstone/assets/46277840/7aa1f322-649b-4aab-9b3d-62930bd224fa">
Run details: 
<img width="1191" alt="Screenshot 2023-11-13 at 13 44 13" src="https://github.com/ttschuemp/nd00333-capstone/assets/46277840/4606a0db-f2ed-4a5b-853b-9bc59fa33461">

<img width="1191" alt="Screenshot 2023-11-13 at 13 44 13" src="https://github.com/ttschuemp/nd00333-capstone/assets/46277840/09582c2f-36fa-4282-9edc-9cf44bb67f86">

Best model detaisl with params:
<img width="1095" alt="Screenshot 2023-11-15 at 16 56 31" src="https://github.com/ttschuemp/nd00333-capstone/assets/46277840/7862fae6-f232-4383-9380-92cd016c2443">


## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

I used a Gradient Boosting Classifier. One compelling argument for the effectiveness of the Gradient Boosting Classifier is its ability to handle complex relationships within the data. This model, through its ensemble of weak learners (typically decision trees), sequentially corrects errors made by preceding models. This iterative learning process allows Gradient Boosting to capture intricate patterns and dependencies in the data, making it particularly powerful for tasks where the relationships are non-linear and exhibit a high degree of complexity. 

Learning Rate Variation:

By default, the learning rate is set to 0.1. However, for flexibility and optimization, we introduce variability by allowing a uniform distribution between 0.1 and 0.5 as a parameterization option.
Number of Base Estimators (n_estimators):

In the Gradient Boosting modeling process, the parameter n_estimators determines the count of base estimators, typically represented by decision trees. To cater to different scenarios, we provide a list of options for this parameter, allowing selection from values such as 100, 200, 300, and 350.


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

The model had an `accuracy of 84%`. The parameter setting that HyperDrive found is: `['earning_rate', '0.29', 'n_estimators', '100']`.

To improve the model we could increase the paramenter space for HyperDrive to search in so for example > 350 estimators might be even better. Another option is to tune even more hyperparameters of the model.

Screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

<img width="1088" alt="Screenshot 2023-11-19 at 13 34 55" src="https://github.com/ttschuemp/nd00333-capstone/assets/46277840/28667a90-318d-46a2-b19b-2e3cd3aa0317">
<img width="1161" alt="Screenshot 2023-11-19 at 13 35 03" src="https://github.com/ttschuemp/nd00333-capstone/assets/46277840/8f33ac8c-860b-434c-9dd6-daae313fc94e">

 Best model trained with it's parameters

 
<img width="1155" alt="Screenshot 2023-11-19 at 13 36 16" src="https://github.com/ttschuemp/nd00333-capstone/assets/46277840/1f04ed38-46c5-40fc-8e38-3b2ec0ea1d58">


 
## Model Deployment

Screenshot of the best model deployment: 

<img width="825" alt="Screenshot 2023-11-19 at 14 06 44" src="https://github.com/ttschuemp/nd00333-capstone/assets/46277840/41ca2d33-191e-414f-b9bd-1ed5514c3b88">

Then I tested with 5 random datapoints the endpoint: 
<img width="729" alt="Screenshot 2023-11-19 at 14 11 55" src="https://github.com/ttschuemp/nd00333-capstone/assets/46277840/8dafd072-4ab2-4290-968a-cc8f2e8f3982">

Here the output of the endpoint:
<img width="962" alt="Screenshot 2023-11-19 at 14 12 11" src="https://github.com/ttschuemp/nd00333-capstone/assets/46277840/373b83f0-b93f-43fb-98f0-03b8508617b0">


## Screen Recording

https://youtu.be/BNxMxUqidCg


## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
