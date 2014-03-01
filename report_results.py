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

def report(path):
    dataset = joblib.load(DATASETS_PATH + "dataset_2_sample_size_1")
    all_pred = pickle.load(open(path))["predictions"]
    
    result = {}
    
    """    
    for key in dataset[0]:
        wrong_records = dataset[0][key]["META_LABEL"]=="label"
        assert np.all(all_pred[key][wrong_records,:]==np.zeros((1,2))), "0 prediction for bad records"
        assert dataset[0][key].shape[0] == all_pred[key].shape[0], "all records wrong"
        
        dataset[0][key] = dataset[0][key].ix[~wrong_records]
        all_pred[key] = all_pred[key][~wrong_records,:]
    """
    
    for key in dataset[0]:    
        validation_set = np.where(dataset[0][key]["META_LABEL"]=="VALIDATION")[0]
        train_set = np.where(dataset[0][key]["META_LABEL"]=="TRAINING")[0]
    
        predictions = np.zeros((len(validation_set)+len(train_set),2))
        for scorer in [1,2]:
            predictions[:,scorer - 1] = all_pred[key][:,scorer-1]
        merged_pred = predictions.mean(axis=1)
        
        predictions_scores_df = pd.DataFrame({'response':np.array(dataset[0][key]["META_SCORE_1"])[train_set],
                                              'predictions': merged_pred[train_set]})
        #predictions_scores_df = predictions_scores_df.ix[~predictions_scores_df["response"].isnull(),:]
        predictions_scores_df["response"] = predictions_scores_df["response"].fillna(0).map(int)
        filter_valid_responses = ~predictions_scores_df["response"].isnull()
        
        train_goc = grade_on_a_curve(merged_pred[train_set], dataset[0][key]["META_SCORE_%d" % scorer].fillna(0)[train_set].map(int))
                    
        kappa_goc = kappa.quadratic_weighted_kappa(train_goc, predictions_scores_df["response"].fillna(0))
        
        result[key] = kappa_goc
    return result
    
def get_human_results():
    dataset = joblib.load(DATASETS_PATH + "dataset_2_sample_size_1")
    result = {}
    for key in dataset[0]:
        filter_records = dataset[0][key]["META_LABEL"]=="TRAINING"
        dataset[0][key] = dataset[0][key].ix[filter_records]

    for key in dataset[0]:
        result[key] = kappa.quadratic_weighted_kappa(dataset[0][key]["META_SCORE_1"].fillna(0).astype(np.int), 
                                                     dataset[0][key]["META_SCORE_2"].fillna(0).astype(np.int))
    return result    
    
def get_model_results():
    output = open("model_results.csv","w")
    output.write("model name;dataset;parameters;essay;score\n")        
    
    for k,v in get_human_results().items():
        output.write("%s;%s;%s;%s;%.4f\n" % ("HUMAN", "-","-", k, v))    
    
    for path in glob.glob("models/*"):
        if os.stat(path).st_size == 0:
            continue
        print path
        all_pred = pickle.load(open(path))
        results = report(path)
        for k,v in results.items():
            parameters = "_".join(["%s=%s" % (k_,v_) for k_,v_ in all_pred["parameters"].items()])
            output.write("%s;%s;%s;%s;%.4f\n" % (all_pred["model_name"], all_pred["dataset"], parameters, k, v))
    output.close()
            

        
if __name__ == "__main__":
    get_model_results()