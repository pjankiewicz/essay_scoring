import glob
import pandas as pd

class Essays():
    def __init__(self,path):
        self.load_essays(path)
        
    def load_essays(self,path):
        essays = {}
        for essaypath in glob.glob(path):
            print essaypath
            essay_name = essaypath.split("/")[-1][:-4]
            df = pd.read_csv(essaypath)
    
            #removes empty scores        
            #df = df.ix[~(df.read_1_score.isnull() | df.read_2_score.isnull()),:].reset_index()
            df["final_score"] = df.read_1_score
            for r in range(df.shape[0]):
                if df.ix[r,"read_1_score"] != df.ix[r,"read_2_score"]:
                    df.ix[r,"final_score"] = df.ix[r,"read_3_score"]
            
            df["data_answer"] = df["data_answer"].map(str)
            
            essays[essay_name] = df
        self.essays = essays
        
    def get_list_of_essays(self):
        return self.essays.keys()
        
    def apply_cell_function(self,col,func,essays=None,exclude=None):
        if essays is None:
            essays = self.get_list_of_essays()
        if exclude is not None:
            essays = [e for e in essays if e not in exclude]
        output = {}
        for k in self.essays:
            output[k] = self.essays[k][col].map(func)
        return output

    def apply_vector_function(self,col,func,essays=None,exclude=None):
        if essays is None:
            essays = self.get_list_of_essays()
        if exclude is not None:
            essays = [e for e in essays if e not in exclude]
        output = {}
        for k in self.essays:
            output[k] = func(self.essays[k][col])
        return output
    
        