import string
import os
import cPickle as pickle

from lib.data_io import Essays
from lib.utils import *
from lib.text_transform import *
from lib.math_helpers import *
from lib.pipeline import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib

from lib.porter import stemmer

def create_dataset_2():
    essays = Essays("data_work/items_data/*.csv")   
    
    LABEL = essays.apply_cell_function("label",identity)
    READ_1_SCORE = essays.apply_cell_function("read_1_score",identity)
    READ_2_SCORE = essays.apply_cell_function("read_2_score",identity)
    FINAL_SCORE = essays.apply_cell_function("final_score",identity)
    
    # prepares text
    print "Preparing text..."
    RAW_TEXTS = essays.apply_cell_function("data_answer",identity)
    RAW_TEXTS = func_over_dict(RAW_TEXTS, apply_map_func(string.upper),parallel=True)
    TEXTS = essays.apply_cell_function("data_answer",identity)
    
    print "Cleaning text..."    
    TEXTS = clean_text(TEXTS)
    
    print "Spellchecking..."
    TEXTS = func_over_dict(TEXTS, apply_map_func(lambda x: spellcheck(x,exclude=["EQUATIONINCORRECT","EQUATIONINCORRECT"])))
    for key in ["5_53299","7_46793","3_51802","7_46597"]:
        TEXTS[key] = TEXTS[key].map(simplify_math)
    
    print "Reducing vocabulary..."
    TEXTS = reduce_vocabulary_dict(TEXTS)
    
    print "Stemming..."
    TEXTS = func_over_dict(TEXTS, apply_map_func(lambda x: " ".join([stemmer(w) for w in x.split()])))
    
    bow_args = {'min_df':2,'ngram_range':(1,1),'stop_words':'english','tokenizer':lambda x: x.split()}
    BOW_1_GRAM       = func_over_dict(TEXTS, lambda x: bag_of_words(x,**bow_args))

    bow_args = {'min_df':5,'ngram_range':(2,2),'stop_words':'english','tokenizer':lambda x: x.split()}
    BOW_2_GRAM       = func_over_dict(TEXTS, lambda x: bag_of_words(x,**bow_args))

    bow_args = {'min_df':5,'ngram_range':(3,3),'stop_words':'english','tokenizer':lambda x: x.split()}
    BOW_3_GRAM       = func_over_dict(TEXTS, lambda x: bag_of_words(x,**bow_args))

    META_LABEL       = func_over_dict(LABEL, lambda x: pd.DataFrame({'META_LABEL':x}))
    META_SCORE_1     = func_over_dict(READ_1_SCORE, lambda x: pd.DataFrame({'META_SCORE_1':x}))
    META_SCORE_2     = func_over_dict(READ_2_SCORE, lambda x: pd.DataFrame({'META_SCORE_2':x}))
    META_SCORE_FINAL = func_over_dict(FINAL_SCORE, lambda x: pd.DataFrame({'META_SCORE_FINAL':x}))
    TEXT_STATISTICS  = func_over_dict(RAW_TEXTS, text_statistics)
    QUOTATIONS_NUM   = func_over_dict(func_over_dict(RAW_TEXTS, apply_map_func(lambda x: len(re.findall('"(.*?)"',x)))), lambda x: pd.DataFrame({'QUOTATIONS_NUM':x}))
    YES_POSITION     = func_over_dict(func_over_dict(RAW_TEXTS, apply_map_func(lambda x: x.find("YES"))), lambda x: pd.DataFrame({'YES_POSITION':x}))
    NO_POSITION      = func_over_dict(func_over_dict(RAW_TEXTS, apply_map_func(lambda x: x.find("NO"))), lambda x: pd.DataFrame({'NO_POSITION':x}))
    
    dataset = merge_dataframes([
        META_LABEL
       ,META_SCORE_1
       ,META_SCORE_2
       ,META_SCORE_FINAL
       ,BOW_1_GRAM
       ,BOW_2_GRAM
       ,BOW_3_GRAM
       ,TEXT_STATISTICS
       ,QUOTATIONS_NUM
       ,YES_POSITION
       ,NO_POSITION
    ])
    
    dataset = [dataset, dataset]

    joblib.dump(dataset,"data_work/datasets/dataset_2_SE")
    

def create_dataset_2_SE():
    essays = Essays("data_work/items_data_se/*.csv")   
    
    LABEL = essays.apply_cell_function("label",identity)
    READ_1_SCORE = essays.apply_cell_function("read_1_score",identity)
    READ_2_SCORE = essays.apply_cell_function("read_2_score",identity)
    FINAL_SCORE = essays.apply_cell_function("final_score",identity)
    
    # prepares text
    print "Preparing text..."
    RAW_TEXTS = essays.apply_cell_function("data_answer",identity)
    RAW_TEXTS = func_over_dict(RAW_TEXTS, apply_map_func(string.upper))
    TEXTS = essays.apply_cell_function("data_answer",identity)
    
    print "Cleaning text..."    
    TEXTS = clean_text(TEXTS)
    
    print "Spellchecking..."
    TEXTS = func_over_dict(TEXTS, apply_map_func(lambda x: spellcheck(x,exclude=["EQUATIONINCORRECT","EQUATIONINCORRECT"])))
    #for key in ["5_53299","7_46793","3_51802","7_46597"]:
    #    TEXTS[key] = TEXTS[key].map(simplify_math)
    
    print "Reducing vocabulary..."
    TEXTS = reduce_vocabulary_dict(TEXTS)
    
    print "Stemming..."
    TEXTS = func_over_dict(TEXTS, apply_map_func(lambda x: " ".join([stemmer(w) for w in x.split()])))
    
    bow_args = {'min_df':2,'ngram_range':(1,1),'stop_words':'english','tokenizer':lambda x: x.split()}
    BOW_1_GRAM       = func_over_dict(TEXTS, lambda x: bag_of_words(x,**bow_args))

    #bow_args = {'min_df':5,'ngram_range':(2,2),'stop_words':'english','tokenizer':lambda x: x.split()}
    #BOW_2_GRAM       = func_over_dict(TEXTS, lambda x: bag_of_words(x,**bow_args))

    #bow_args = {'min_df':5,'ngram_range':(3,3),'stop_words':'english','tokenizer':lambda x: x.split()}
    #BOW_3_GRAM       = func_over_dict(TEXTS, lambda x: bag_of_words(x,**bow_args))

    META_LABEL       = func_over_dict(LABEL, lambda x: pd.DataFrame({'META_LABEL':x}))
    META_SCORE_1     = func_over_dict(READ_1_SCORE, lambda x: pd.DataFrame({'META_SCORE_1':x}))
    META_SCORE_2     = func_over_dict(READ_2_SCORE, lambda x: pd.DataFrame({'META_SCORE_2':x}))
    META_SCORE_FINAL = func_over_dict(FINAL_SCORE, lambda x: pd.DataFrame({'META_SCORE_FINAL':x}))
    TEXT_STATISTICS  = func_over_dict(RAW_TEXTS, text_statistics)
    QUOTATIONS_NUM   = func_over_dict(func_over_dict(RAW_TEXTS, apply_map_func(lambda x: len(re.findall('"(.*?)"',x)))), lambda x: pd.DataFrame({'QUOTATIONS_NUM':x}))
    YES_POSITION     = func_over_dict(func_over_dict(RAW_TEXTS, apply_map_func(lambda x: x.find("YES"))), lambda x: pd.DataFrame({'YES_POSITION':x}))
    NO_POSITION      = func_over_dict(func_over_dict(RAW_TEXTS, apply_map_func(lambda x: x.find("NO"))), lambda x: pd.DataFrame({'NO_POSITION':x}))
    
    dataset = merge_dataframes([
        META_LABEL
       ,META_SCORE_1
       ,META_SCORE_2
       ,META_SCORE_FINAL
       ,BOW_1_GRAM
       #,BOW_2_GRAM
       #,BOW_3_GRAM
       ,TEXT_STATISTICS
       ,QUOTATIONS_NUM
       ,YES_POSITION
       ,NO_POSITION
    ])
    
    dataset = [dataset, dataset]

    joblib.dump(dataset,"data_work/datasets/dataset_2_SE")
    

def create_dataset_2_gaming():
    essays = Essays("data_work/items_data_gaming/*.csv")   
    
    LABEL = essays.apply_cell_function("label",identity)
    READ_1_SCORE = essays.apply_cell_function("read_1_score",identity)
    READ_2_SCORE = essays.apply_cell_function("read_2_score",identity)
    FINAL_SCORE = essays.apply_cell_function("final_score",identity)
    
    # prepares text
    print "Preparing text..."
    RAW_TEXTS = essays.apply_cell_function("data_answer",identity)
    RAW_TEXTS = func_over_dict(RAW_TEXTS, apply_map_func(string.upper))
    TEXTS = essays.apply_cell_function("data_answer",identity)
    
    print "Cleaning text..."    
    TEXTS = clean_text(TEXTS)
    
    print "Spellchecking..."
    TEXTS = func_over_dict(TEXTS, apply_map_func(lambda x: spellcheck(x,exclude=["EQUATIONINCORRECT","EQUATIONINCORRECT"])))
    #for key in ["5_53299","7_46793","3_51802","7_46597"]:
    #    TEXTS[key] = TEXTS[key].map(simplify_math)
    
    print "Reducing vocabulary..."
    TEXTS = reduce_vocabulary_dict(TEXTS)
    
    print "Stemming..."
    TEXTS = func_over_dict(TEXTS, apply_map_func(lambda x: " ".join([stemmer(w) for w in x.split()])))
    
    bow_args = {'min_df':2,'ngram_range':(1,1),'stop_words':'english','tokenizer':lambda x: x.split()}
    BOW_1_GRAM       = func_over_dict(TEXTS, lambda x: bag_of_words(x,**bow_args))

    #bow_args = {'min_df':5,'ngram_range':(2,2),'stop_words':'english','tokenizer':lambda x: x.split()}
    #BOW_2_GRAM       = func_over_dict(TEXTS, lambda x: bag_of_words(x,**bow_args))

    #bow_args = {'min_df':5,'ngram_range':(3,3),'stop_words':'english','tokenizer':lambda x: x.split()}
    #BOW_3_GRAM       = func_over_dict(TEXTS, lambda x: bag_of_words(x,**bow_args))

    META_LABEL       = func_over_dict(LABEL, lambda x: pd.DataFrame({'META_LABEL':x}))
    META_SCORE_1     = func_over_dict(READ_1_SCORE, lambda x: pd.DataFrame({'META_SCORE_1':x}))
    META_SCORE_2     = func_over_dict(READ_2_SCORE, lambda x: pd.DataFrame({'META_SCORE_2':x}))
    META_SCORE_FINAL = func_over_dict(FINAL_SCORE, lambda x: pd.DataFrame({'META_SCORE_FINAL':x}))
    TEXT_STATISTICS  = func_over_dict(RAW_TEXTS, text_statistics)
    QUOTATIONS_NUM   = func_over_dict(func_over_dict(RAW_TEXTS, apply_map_func(lambda x: len(re.findall('"(.*?)"',x)))), lambda x: pd.DataFrame({'QUOTATIONS_NUM':x}))
    YES_POSITION     = func_over_dict(func_over_dict(RAW_TEXTS, apply_map_func(lambda x: x.find("YES"))), lambda x: pd.DataFrame({'YES_POSITION':x}))
    NO_POSITION      = func_over_dict(func_over_dict(RAW_TEXTS, apply_map_func(lambda x: x.find("NO"))), lambda x: pd.DataFrame({'NO_POSITION':x}))
    
    dataset = merge_dataframes([
        META_LABEL
       ,META_SCORE_1
       ,META_SCORE_2
       ,META_SCORE_FINAL
       ,BOW_1_GRAM
       #,BOW_2_GRAM
       #,BOW_3_GRAM
       ,TEXT_STATISTICS
       ,QUOTATIONS_NUM
       ,YES_POSITION
       ,NO_POSITION
    ])
    
    dataset = [dataset, dataset]

    joblib.dump(dataset,"data_work/datasets/dataset_2_gaming")    

def create_dataset_2_sample_size(input_path, output_path):
    essays = Essays(input_path)  
    
    LABEL = essays.apply_cell_function("label",identity)
    READ_1_SCORE = essays.apply_cell_function("read_1_score",identity)
    READ_2_SCORE = essays.apply_cell_function("read_2_score",identity)
    FINAL_SCORE = essays.apply_cell_function("final_score",identity)
    
    # prepares text
    print "Preparing text..."
    RAW_TEXTS = essays.apply_cell_function("data_answer",identity)
    RAW_TEXTS = func_over_dict(RAW_TEXTS, apply_map_func(string.upper), parallel=True)
    TEXTS = essays.apply_cell_function("data_answer",identity)
    
    print "Cleaning text..."    
    TEXTS = clean_text(TEXTS)
    
    print "Simplifying math expressions..."
    math_essays = [key for key in TEXTS.keys() if key.startswith("53299")]
    for key in math_essays:
        TEXTS[key] = TEXTS[key].map(simplify_math)
    
    print "Spellchecking..."
    TEXTS = func_over_dict(TEXTS, apply_map_func(spellcheck), parallel=False)    
    
    print "Reducing vocabulary..."
    TEXTS = func_over_dict(TEXTS, reduce_vocabulary_func, parallel=True)
    
    print "Stemming..."
    TEXTS = func_over_dict(TEXTS, apply_map_func(lambda x: " ".join([stemmer(w) for w in x.split()])), parallel=True)
    
    bow_args = {'min_df':2,'ngram_range':(1,1),'stop_words':'english','tokenizer':lambda x: x.split()}
    BOW_1_GRAM       = func_over_dict(TEXTS, lambda x: bag_of_words(x,**bow_args), parallel=True)

    META_LABEL       = func_over_dict(LABEL, lambda x: pd.DataFrame({'META_LABEL':x}), parallel=True)
    META_SCORE_1     = func_over_dict(READ_1_SCORE, lambda x: pd.DataFrame({'META_SCORE_1':x}), parallel=True)
    META_SCORE_2     = func_over_dict(READ_2_SCORE, lambda x: pd.DataFrame({'META_SCORE_2':x}), parallel=True)
    META_SCORE_FINAL = func_over_dict(FINAL_SCORE, lambda x: pd.DataFrame({'META_SCORE_FINAL':x}), parallel=True)
    TEXT_STATISTICS  = func_over_dict(RAW_TEXTS, text_statistics, parallel=True)
    QUOTATIONS_NUM   = func_over_dict(func_over_dict(RAW_TEXTS, apply_map_func(lambda x: len(re.findall('"(.*?)"',x)))), lambda x: pd.DataFrame({'QUOTATIONS_NUM':x}), parallel=True)
    YES_POSITION     = func_over_dict(func_over_dict(RAW_TEXTS, apply_map_func(lambda x: x.find("YES"))), lambda x: pd.DataFrame({'YES_POSITION':x}), parallel=True)
    NO_POSITION      = func_over_dict(func_over_dict(RAW_TEXTS, apply_map_func(lambda x: x.find("NO"))), lambda x: pd.DataFrame({'NO_POSITION':x}), parallel=True)
    
    dataset = merge_dataframes([
        META_LABEL
       ,META_SCORE_1
       ,META_SCORE_2
       ,META_SCORE_FINAL
       ,BOW_1_GRAM
       ,TEXT_STATISTICS
       ,QUOTATIONS_NUM
       ,YES_POSITION
       ,NO_POSITION
    ])
    
    dataset = [dataset, dataset]

    joblib.dump(dataset,output_path)
    
if __name__=="__main__":
    create_dataset_2()
    create_dataset_2_SE()
    create_dataset_2_gaming()
    create_dataset_2_sample_size("data_work/items_datasample_size/*.csv" , "data_work/datasets/dataset_2_sample_size")
