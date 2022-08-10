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

    ds = TabularDatasetFactory.from_delimited_files(path='https://storage.googleapis.com/kagglesdsdata/datasets/2346686/3954047/scores.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220809%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220809T152449Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=376ed0e9e7c679ea63fde67eb10bbeb8f12d462521a4e330ff5a3e8eaaa6700fbc4a5f24bef2cb662ae4c51f709b60ed8bcbf2dc326bfe006d69d725d63af142b4ceb7413ed6f8134c611e62da4ae6f02adef5a6f8208131dcfdac4de7cc86b6c6fdc10b88f3b30a22ffe2fcb2b33a5ba4d1e70635772990121f6c06625d18e936cd77942df59d72422b261f67bcc6a185b3c002d2e90e1174faf9d10ba066bfc515a31754344c112772754656de7fbca9f9f46d8dcba1e37342793e9b79f35c29ec3c723d720be26c1b6b1bbb6601bf22c84b063ebc1e975e1f00179cea6214cb96ac6173f5b7bf3d7009b6d8986003a7c25fa022bc1795c9e2de70879d0536', separator=',')
    
    x, y = clean_data(ds)

    # TODO: Split data into train and test sets.

    ### YOUR CODE HERE ###a
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()