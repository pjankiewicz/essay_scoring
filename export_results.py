# -*- coding: utf-8 -*-
import cPickle as pickle

import numpy as np
from sklearn.externals import joblib

from config import *
from lib.utils import *
from lib import kappa
from lib.data_io import Essays

dataset = joblib.load(DATASETS_PATH + "dataset_2_sample_size")
all_pred = pickle.load(open(MODELS_PATH + "ENSEMBLE_dataset_ensemble_sample_size_2590e6e68e14f60ee12bdbd0887fbafb"))["predictions"]
essays = Essays("data_work/items_data_sample_size/*.csv")

result = {}
for key in dataset[0]:
    wrong_records = dataset[0][key]["META_LABEL"]=="label"
    assert np.all(all_pred[key][wrong_records,:]==np.zeros((1,2))), "0 prediction for bad records"
    assert dataset[0][key].shape[0] == all_pred[key].shape[0] == essays.essays[key].shape[0], "all records wrong"
    assert all_pred[key].shape[0] == essays.essays[key].shape[0], "all records wrong"
    
    dataset[0][key] = dataset[0][key].ix[~wrong_records]
    all_pred[key] = all_pred[key][~wrong_records,:]
    essays.essays[key] = essays.essays[key].ix[~wrong_records,:]


for key in dataset[0]:    
    validation_set = np.where(dataset[0][key]["META_LABEL"]=="VALIDATION")[0]
    train_set = np.where(dataset[0][key]["META_LABEL"]=="TRAINING")[0]

    predictions = np.zeros((len(validation_set)+len(train_set),2))
    for scorer in [1,2]:
        predictions[:,scorer - 1] = all_pred[key][:,scorer-1]
    merged_pred = predictions.mean(axis=1)
    
    predictions_scores_df = pd.DataFrame({'response':np.array(dataset[0][key]["META_SCORE_1"])[train_set],
                                          'predictions': merged_pred[train_set]})
    predictions_scores_df["response"][predictions_scores_df["response"].isnull()] = 0
    predictions_scores_df["response"] = predictions_scores_df["response"].map(int)
    
    train_goc = grade_on_a_curve(predictions_scores_df["predictions"], predictions_scores_df["response"]).astype(np.int)
    
    kappa_goc = kappa.quadratic_weighted_kappa(train_goc, predictions_scores_df["response"])
    kappa_raw = kappa.quadratic_weighted_kappa(predictions_scores_df["predictions"].round().map(int), predictions_scores_df["response"])
    
    print key, kappa_goc, kappa_raw
    
    pred_goc = grade_on_a_curve(merged_pred[validation_set], predictions_scores_df["response"]).astype(np.int)    
    print set(pred_goc), set(predictions_scores_df["response"])    
    assert set(pred_goc)==set(predictions_scores_df["response"]), "wrong set of values"
    
    result[key] = pred_goc #kappa.quadratic_weighted_kappa(pred_goc, dataset[0][key]["META_SCORE_FINAL"].fillna(0).astype(np.int))


"""
<?xml version="1.0" encoding="UTF-8"?>
<Job_Details xmlns="http://www.imsglobal.org/xsd/imscp_v1p1" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="ctb_score.xsd" Score_Provider_Name="AI-XX" Case_Count="10" Date_Time="20130815160834">
   <Student_Details Vendor_Student_ID="361228">
      <Student_Test_List>
         <Student_Test_Details Student_Test_ID="3129362" Grade="8" Total_CR_Item_Count="1">
            <Item_DataPoint_List>
               <Item_DataPoint_Details Item_ID="12345" Data_Point="" Item_No="1" Final_Score="0">
                  <Read_Details Read_Number="1" Score_Value="0" Reader_ID="490" Date_Time="20131206134100" />
               </Item_DataPoint_Details>
            </Item_DataPoint_List>
         </Student_Test_Details>
      </Student_Test_List>
   </Student_Details>
</Job_Details>
"""

ALL_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Job_Details xmlns="http://www.imsglobal.org/xsd/imscp_v1p1" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="ctb_score.xsd" Score_Provider_Name="AI-XX" Case_Count="10" Date_Time="20130815160834">
%s</Job_Details>
"""

ITEM_XML = """   <Student_Details Vendor_Student_ID="%s">
      <Student_Test_List>
         <Student_Test_Details Student_Test_ID="%s" Grade="%s" Total_CR_Item_Count="1">
            <Item_DataPoint_List>
               <Item_DataPoint_Details Item_ID="%s" Data_Point="" Item_No="1" Final_Score="%d">
                  <Read_Details Read_Number="1" Score_Value="%d" Reader_ID="1" Date_Time="20131206134100" />
               </Item_DataPoint_Details>
            </Item_DataPoint_List>
         </Student_Test_Details>
      </Student_Test_List>
   </Student_Details>
"""

for key in dataset[0]:
    validation_set = dataset[0][key]["META_LABEL"]=="VALIDATION"
    train_set = dataset[0][key]["META_LABEL"]=="TRAINING"

    ess = essays.essays[key].ix[validation_set,:].reset_index().drop('index',axis=1)
    res = result[key]
    assert len(ess) == len(res)
    item_id = key #[-5:]
    
    items = []
    for n in range(len(ess)):
        item = ITEM_XML % ((ess["data_meta_vendor_student_id"][n]),
                           (ess["data_meta_student_test_id"][n]),
                           (ess["data_meta_student_grade"][n]),
                           (item_id),
                           int(res[n]),
                           int(res[n]))
        items.append(item)
        
    out = open(FINAL_SCORES + key + "_AI-PJ_scores.xml","w")
    out.write(ALL_XML % ("".join(items)))
    out.close()
