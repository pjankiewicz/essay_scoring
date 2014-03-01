from multiprocessing import Pool
import cPickle as pickle
import hashlib

import os

import pandas as pd

# models used
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn import linear_model

from sklearn.externals import joblib

from lib.utils import *
from lib import kappa

from config import *

from sklearn import cross_validation

os.system("taskset -p 0xff %d" % os.getpid())

def cv(args):
    kf,ds,model,n,scorer = args
    cols = [col for col in ds.columns if not col.startswith("META_")]
    X_tr = np.array(ds[cols])[kf[n][0],:]
    y_tr = ds["META_SCORE_%d" % (scorer)].fillna(0)[kf[n][0]].map(int)
    X_te = np.array(ds[cols])[kf[n][1],:]
    y_te = ds["META_SCORE_%d" % (scorer)].fillna(0)[kf[n][1]].map(int)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return pred

def get_md5_hash(s):
    m = hashlib.md5(s)
    return m.hexdigest()

def touch(path):
    a = open(path, 'a').close()

def create_model(dataset_name, model, model_name, overwrite=False,parallel=True):
    args_string = "_".join(["%s=%s" % (k,v) for k,v in model.get_params().items()])
    args_hash = get_md5_hash(args_string)
    filename = MODELS_PATH + model_name + "_" + dataset_name + "_" + args_hash
    
    print "Creating", filename
    if os.path.exists(filename) and not overwrite:
        print "It exists so skipping..."
        return
        
    # touch file so that other processes don't recreate it
    touch(filename)
        
    dataset = joblib.load(DATASETS_PATH + dataset_name)    
    all_pred = {}    

    n_datasets = len(dataset[0].keys())
    # for each essay
    for i, key in enumerate(dataset[0]):
        print "Creating %d of %d" % (i+1, n_datasets)
        predictions = np.zeros((dataset[0][key].shape[0],2))
        trainset = np.where(dataset[0][key]["META_LABEL"] == "TRAINING")[0]
        testset = np.where(dataset[0][key]["META_LABEL"] == "VALIDATION")[0]
        print "Training #", len(trainset), "Testing #", len(testset)
        print "Shape", dataset[0][key].shape
        # 2 datasets - each for 1 scorer
        for scorer in [1,2]:
            ds = dataset[scorer - 1][key]
            kf = cross_validation.KFold(len(trainset), n_folds=7)
            kf = [(trainset[tr], trainset[te]) for tr, te in kf] + [(trainset,testset)]
            pred = np.zeros(ds.shape[0])         
            if parallel:
                pool = Pool(processes=8)
                essay_sets = pool.map(cv, [[kf, ds, model, n, scorer] for n in range(8)])
                pool.close()

                for n, essay_set in enumerate(essay_sets):
                    pred[kf[n][1]] = essay_set
            else:
                for n in range(8):
                    pred[kf[n][1]] = cv([kf,ds,model,n,scorer])

            predictions[:,scorer - 1] = pred

            # DEBUG
            pred_goc = grade_on_a_curve(pred[trainset], ds["META_SCORE_%d" % scorer].fillna(0)[trainset].map(int))
            print key, scorer, kappa.quadratic_weighted_kappa(pred_goc, ds["META_SCORE_%d" % (scorer)].fillna(0)[trainset].map(int))

        #DEBUG
        merged_pred = predictions.mean(axis=1)
        pred_goc = grade_on_a_curve(merged_pred[trainset], ds["META_SCORE_%d" % scorer].fillna(0)[trainset].map(int))
        print key, "Merged", kappa.quadratic_weighted_kappa(pred_goc, ds["META_SCORE_%d" % (scorer)].fillna(0)[trainset].map(int))
        print 
 
        all_pred[key] = predictions
    
    data = {"predictions":all_pred
            ,"model_name":model_name
            ,"dataset":dataset_name
            ,"parameters":model.get_params()}
    pickle.dump(data, open(filename,"w"))
    
def report_results(path):
    all_pred = pickle.load(open(path))["predictions"]
    for ds in dataset:
        for key in ds:
            predictions = all_pred[key]
            merged_pred = predictions.mean(axis=1)
            pred_goc = grade_on_a_curve(merged_pred, ds[key]["META_SCORE_%d" % scorer].fillna(0)).astype(np.int)
            print key, "FINAL", kappa.quadratic_weighted_kappa(pred_goc, ds[key]["META_SCORE_FINAL"].fillna(0).astype(np.int))
            print 
            
            
if __name__ == "__main__":
    # GBM 
    model_name = "GBM"
    for n_estimators in [500,1000,1500]:
        for max_depth in [3,4,5,7,9,11]:
            for max_features in [0.5, None]:
                for dataset_name in ["dataset_2_SE"]:
                    #assert False
                    print n_estimators, max_depth, max_features
                    model = GradientBoostingRegressor(n_estimators=n_estimators,
                                                      learning_rate=0.01,
                                                      subsample=0.5,
                                                      max_depth=max_depth, 
                                                      max_features=max_features,
                                                      random_state=0, 
                                                      loss='ls')
                    
                    create_model(dataset_name=dataset_name,
                                 model=model,
                                 model_name=model_name)
