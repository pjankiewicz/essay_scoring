# -*- coding: utf-8 -*-
import os
import glob
import cPickle as pickle

import pandas as pd

from sklearn.externals import joblib
from lib.utils import *
from lib import kappa

MODELS_PATH = "models/"
DATASETS_PATH = "data_work/datasets/"

def single_model_pred(path):
    all_pred = pickle.load(open(path))
    parameters = all_pred["model_name"] + "_" + "_".join(["%s=%s" % (k_,v_) for k_,v_ in all_pred["parameters"].items()])
    result = {}
    for key in all_pred["predictions"]:
        predictions = np.zeros((all_pred["predictions"][key].shape[0],2))
        for scorer in [1,2]:
            predictions[:,scorer - 1] = all_pred["predictions"][key][:,scorer-1]
        merged_pred = predictions.mean(axis=1)
        result[key] = merged_pred
    return result, parameters

def get_model_preds():
    dataset = joblib.load(DATASETS_PATH + "dataset_2_sample_size_1")
    predictions = {}
    for path in glob.glob("models/*"):
        print path
        if os.stat(path).st_size == 0:
            continue
        all_pred = pickle.load(open(path))
        pred, model_name = single_model_pred(path)
        for k,v in pred.items():
            if k not in predictions:
                ds = dataset[0][k][["META_LABEL","META_SCORE_1","META_SCORE_2","META_SCORE_FINAL"]]
                predictions[k] = ds
            predictions[k][model_name] = v
    final_dataset = [predictions,predictions]
    joblib.dump(final_dataset,"data_work/datasets/dataset_ensemble_sample_size")
            
           
if __name__ == "__main__":
    get_model_preds()