import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb


def init():
    global bst
    model_root = os.getenv("AZUREML_MODEL_DIR")
    # The name of the folder in which to look for LightGBM model files
    #lgbm_model_folder = "model"
    lgbm_model_folder = "user_logs"
    bst = lgb.Booster(
        model_file=os.path.join(model_root, lgbm_model_folder, "bst-model.txt")
    )


def run(raw_data):
    columns = bst.feature_name()
    data = np.array(json.loads(raw_data)["data"])
    test_df = pd.DataFrame(data=data, columns=columns)
    # Make prediction
    out = bst.predict(test_df)
    return out.tolist()