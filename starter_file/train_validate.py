from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core import Dataset, Datastore
from azureml.data.dataset_factory import TabularDatasetFactory

def clean_data(data):
    # Dict for cleaning data

    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    x_df['Student Placed'] = x_df['Student Placed'].replace(['Yes'],1)
    x_df['Student Placed'] = x_df['Student Placed'].replace(['No'],0)
    
    x = x_df.drop('Student Placed',axis=1).values
    y = x_df['Student Placed'].values
    return x, y

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    # TODO: Create TabularDataset using TabularDatasetFactory
    # Data is located at:
    # "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

    ds = TabularDatasetFactory.from_delimited_files(path='https://storage.googleapis.com/kagglesdsdata/datasets/2346686/3954047/scores.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220814%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220814T141348Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=8d4e92f82985b57fbf6bf6dc6de47d7a1a2d98320647ae81c0f33498e558759c90f61c40a32eb693be50bb47f79ebdf6ba7c9a5024d24234642ea96ef4f80fea510080203feefd8f2712231790a2229377886f6ebc4aa3758d722cfb2728fb8eec984dfe488faeb12b0072d05cb856ed643621e25729074094b56fa116648c2bce78d7f8ecf77226e0817a78d5d5c888f3c539c8ab369fc638f540b53b29dd5e7dba8d54602977895a40ab67efacdb2c5d5f8b574a3c1e225fa83751cab1d47dc294ed94201240fc07b3d385411aaf08217947d33cd1b781c9c267b33cd332d75cd6ffee4efabe17e92135ffabf5ee3e1d754f03cf65cc5c89e01745f8eb0bfc', separator=',')
    
    x, y = clean_data(ds)

    # TODO: Split data into train and test sets.

    ### YOUR CODE HERE ###a
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()