MAIN_PATH = "/home/pawel/SpiderOak Hive/McGraw"

# defines columns to output xml -> csv files
READINGS_LIST = [1,2,3,4,5]
COL_NAMES = ["label","data_meta_ethnicity","data_meta_iep","data_meta_lep","data_meta_gender","data_meta_vendor_student_id","data_meta_student_grade","data_meta_student_test_id","data_answer"]
COL_NAMES.extend(["read_%d_reader_id" % (k) for k in READINGS_LIST])
COL_NAMES.extend(["read_%d_score" % (k) for k in READINGS_LIST])
COL_NAMES.extend(["read_%d_condition_code" % (k) for k in READINGS_LIST])

DATA_POINTS = ["A","B","C"]

MODELS_PATH = "models/"
DATASETS_PATH = "data_work/datasets/"
FINAL_SCORES = "data_work/scores/"
