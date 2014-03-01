from __future__ import division

import os
import zipfile
import string
import glob
import csv
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np

from sklearn.externals import joblib

import multiprocessing
from itertools import izip
from parallel_map import parallel_map

def spawn(f):
    def fun(q_in,q_out):
        while True:
            i,x = q_in.get()
            if i is None:
                break
            q_out.put((i,f(x)))
    return fun

def parmap(f, X, nprocs = multiprocessing.cpu_count()):
    q_in   = multiprocessing.Queue(1)
    q_out  = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=spawn(f),args=(q_in,q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i,x)) for i,x in enumerate(X)]
    [q_in.put((None,None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i,x in sorted(res)]   


def convert_xml_to_csv(inp,out,col_names,readings_list,append=False,data_point=None,filezipped=True):
    if append:
        output = open(out,"a")
    else:
        output = open(out,"w")
    csvwriter = csv.writer(output, delimiter=',',quotechar='"')
    if not append:
        csvwriter.writerow(col_names)
    
    if filezipped:
        with zipfile.ZipFile(inp) as myzip:
            xmldata = myzip.read(myzip.namelist()[0])
            root = ET.fromstring(xmldata)
    else:
        with open(inp) as fp:
            xmldata = fp.read()
            root = ET.fromstring(xmldata)
    
    students = root[0][0][0]
    for student in students:
        data_meta_ethnicity = student.attrib['Ethnicity']
        data_meta_iep = student.attrib['IEP']
        data_meta_lep = student.attrib['LEP']
        data_meta_gender = student.attrib['Gender']
        data_meta_vendor_student_id = student.attrib['Vendor_Student_ID']

        student_details = student.findall("./Student_Test_List/Student_Test_Details")[0]
        data_meta_student_grade = student_details.attrib['Grade']
        data_meta_student_test_id = student_details.attrib['Student_Test_ID']
        test = student.findall("./Student_Test_List/Student_Test_Details/Item_List[0]/Item_Details")[0]
        
        data_answer = test.findall("./Item_Response")[0].text.replace("\n","").strip()
        
        scores_reader_id = ['']*len(readings_list)
        scores_values = ['']*len(readings_list)
        scores_condition_codes = ['']*len(readings_list)
        
        try:
            if data_point is None:
                data_final_score = test.findall("./Item_Score_Details/Item_DataPoint_Score_Details")[0].attrib["Final_Score"]
            else:
                data_final_score = test.findall("./Item_Score_Details/Item_DataPoint_Score_Details[@Data_Point='%s']" % (data_point))[0].attrib["Final_Score"]
            for i, reading_id in enumerate(readings_list):
                if data_point is None:
                    score = test.findall("./Item_Score_Details/Item_DataPoint_Score_Details/Score[@Read_Number='%d']" % (reading_id))
                else:
                    score = test.findall("./Item_Score_Details/Item_DataPoint_Score_Details[@Data_Point='%s']/Score[@Read_Number='%d']" % (data_point,reading_id))
                if len(score) > 0:
                    scores_reader_id[i] = score[0].attrib['Reader_ID']
                    if 'Score_Value' in score[0].attrib:
                        scores_values[i] = score[0].attrib['Score_Value']
                    elif 'Condition_Code' in score[0].attrib:
                        scores_condition_codes[i] = score[0].attrib['Condition_Code']
                    else:
                        raise ValueError, "No score or condition code"
        except:
            pass
        
        label = "VALIDATION" if append else "TRAINING"
        row = [label,data_meta_ethnicity,data_meta_iep,data_meta_lep,data_meta_gender,data_meta_vendor_student_id,data_meta_student_grade,data_meta_student_test_id,data_answer]
        row.extend(scores_reader_id)
        row.extend(scores_values)
        row.extend(scores_condition_codes)
        csvwriter.writerow(row)
    output.close()    

def convert_to_binary(df,remove_useless_variables=True):
    # empty
    new_df = pd.DataFrame({"row_id": range(df.shape[0])})   
    new_df.reset_index()
    df["row_id"] = range(df.shape[0])
    cols = list(df.columns)
    column_names = []
    categories = []
    for col in cols:
        unique_values = np.unique(df[col])
        if remove_useless_variables and (len(unique_values) == 1 or col == "row_id"):
            pass
        elif col.startswith("META_"):
            column_name = col
            new_df[column_name] = df[col]
            column_names.append(column_name)   
            categories.append(col)
        elif len(unique_values) == 2:
            # binarne wartosci
            values = df[col]
            values_max = values.max()
            values = values.map(lambda x: 1 if x == values_max else 0).astype('int')
            column_name = "BIN_" + col
            new_df[column_name] = values
            column_names.append(column_name)
            categories.append(col)
        elif df[col].dtype == "object":
            df["count"] = 1
            dfp = df.pivot(index="row_id",columns=col,values="count").fillna(0).astype('int')
            new_column_names = ["CAT_%s_%s" % (col,c) for c in dfp.columns]
            dfp.columns = new_column_names
            new_df = pd.concat([new_df,dfp],axis=1)
            column_names.extend(new_column_names)
            categories.extend([col]*dfp.shape[1])
        else:
            column_name = "NUM_" + col
            new_df[column_name] = df[col]
            column_names.append(column_name)            
            categories.append(col)
        
    new_df = new_df.drop("row_id",axis=1)
    new_df.columns = column_names
    new_df.categories = categories
    return new_df   
   
def func_over_dict(d, func, parallel = False, include=None, exclude=None):
    if include == None:
        keys = d.keys()
    else:
        keys = include
    if exclude is not None:
        keys = [k for k in keys if k not in exclude]
    output = {}
    
    if parallel:
        results = parallel_map(func, [d[key] for key in keys])
        for i, key in enumerate(keys):
            print key            
            output[key] = results[i]
    else:
        for i, key in enumerate(keys):
            print key
            output[key] = func(d[key])
    return output

    
identity = lambda x: x
def apply_map_func(f):
    def helper(x):
        #pd.core.series.Series()
        # old        
        #return x.map(f)
        result = pd.core.series.Series(parmap(f,x))
        return result
    return helper
    
def merge_dataframes(l):
    keys = list(set([item for sublist in l for item in sublist]))
    dfs = {}
    for key in keys:
        print key
        df = None
        for subdf in l:
            if key in subdf:
                if df is None:
                    df = pd.DataFrame(subdf[key])
                else:
                    df = pd.concat([df, subdf[key]],axis=1)
        dfs[key] = df
    return dfs

def remove_empty_responses(y1,X1,y2,X2,fs):
    selection = list(~(y1.isnull() | y2.isnull()))
    y1 = y1[selection]
    X1 = X1.ix[selection,:].reset_index()
    y2 = y2[selection]
    X2 = X2.ix[selection,:].reset_index()
    fs = fs[selection]
    if "level_0" in X1.columns:
        X1 = X1.drop("level_0",axis=1)
    if "level_0" in X2.columns:
        X2 = X2.drop("level_0",axis=1)
    if "index" in X1.columns:
        X1 = X1.drop("index",axis=1)
    if "index" in X1.columns:
        X2 = X2.drop("index",axis=1)
    return (y1,X1,y2,X2,fs)

def remove_empty_responses_dict(y1d,X1d,y2d,X2d,fsd):
    for key in y1d:
        y1d[key], X1d[key], y2d[key], X2d[key], fsd[key] = remove_empty_responses(y1d[key],
                                                                                  X1d[key],
                                                                                  y2d[key],
                                                                                  X2d[key],
                                                                                  fsd[key])
    return (y1d,X1d,y2d,X2d,fsd)
    
def grade_on_a_curve(predictions, ratings):
    #ratings = y_train_1
    #predictions = y_pred
    
    df = pd.DataFrame({"ratings":ratings}).groupby("ratings").count()
    df["ratings_perc"] = df["ratings"] / (len(ratings)*1.0)
    df["cumm"] = df["ratings"].cumsum()
    df["cumm_perc"] = df["cumm"] / (len(ratings)*1.0)
    
    cuts = [predictions.min()] + list(np.percentile(predictions, list(df["cumm_perc"]*100)))
    df_predictions = pd.DataFrame({"predictions":predictions})
    df_predictions["score"] = 0
    for n in range(len(set(ratings))):
        v1 = df_predictions["predictions"] >= cuts[n]
        v2 = df_predictions["predictions"] <= cuts[n+1]
        df_predictions.ix[v1 & v2,"score"] = n
        
    return df_predictions["score"]
    
def read_word2vec_vectors():
    if os.path.exists("/home/pawel/SpiderOak Hive/McGraw/engine/data_work/word2vec_vectors.job"):
        wordvectors_selected = joblib.load("/home/pawel/SpiderOak Hive/McGraw/engine/data_work/word2vec_vectors.job")
    else:
        wordvectors = pd.read_csv("/media/pawel/3c75f1ca-e887-4c5d-b8fb-1d14f69f4163/Klienci/McGraw/corpus/vectors_size_200.txt",sep=" ",header=None)
        vocab = dict([(w,i) for i,w in enumerate(wordvectors.ix[:,0])])

        vocabulary = pd.read_csv("/home/pawel/Dropbox/Klienci/McGraw/engine/vocabulary.txt")
        vocabulary["word"] = vocabulary["word"].map(string.upper)    
        wordvectors_selected = np.zeros((vocabulary.shape[0], 200))
        for i, word in enumerate(vocabulary["word"]):
            print word
            try:
                wordvectors_selected[i,:] = wordvectors.ix[vocab[str(word).upper()],1:200]
            except KeyError:
                pass
        wordvectors_selected = pd.DataFrame(wordvectors_selected)
        wordvectors_selected.index = list(vocabulary["word"])
        joblib.dump(wordvectors_selected,"/home/pawel/Dropbox/Klienci/McGraw/engine/data_work/word2vec_vectors.job")
    
    return wordvectors_selected


