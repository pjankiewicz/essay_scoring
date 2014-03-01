import re
import pandas as pd
import string
import pyaspell

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.externals import joblib 
from sklearn.decomposition import PCA, RandomizedPCA

from utils import *
   
class memoize(dict):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        return self[args]

    def __missing__(self, key):
        result = self[key] = self.func(*key)
        return result    
      
def text_statistics(texts):
    return pd.DataFrame({
     'nword' : texts.map(lambda x: x.count(" "))
    ,'textlength': texts.map(lambda x: len(x))
    ,'nwords_len_1' : texts.map(lambda x: len([w for w in x.split() if len(w) == 1]))
    ,'nwords_len_2' : texts.map(lambda x: len([w for w in x.split() if len(w) == 2]))
    ,'nwords_len_3' : texts.map(lambda x: len([w for w in x.split() if len(w) == 3]))
    ,'nwords_len_4_5' : texts.map(lambda x: len([w for w in x.split() if len(w) in (4,5)]))
    ,'nwords_len_6_7' : texts.map(lambda x: len([w for w in x.split() if len(w) in (6,7)]))
    ,'nwords_len_8_9' : texts.map(lambda x: len([w for w in x.split() if len(w) in (8,9)]))
    ,'nwords_len_10plus' : texts.map(lambda x: len([w for w in x.split() if len(w) >= 10]))
    ,'unique_words' : texts.map(lambda x: len(set(x.split())))
    ,'ndigits' : texts.map(lambda x: len([w for w in x if w in list(string.digits)]))
    ,'symbol>' : texts.map(lambda x: x.count(">"))
    ,'symbol<' : texts.map(lambda x: x.count("<"))
    ,'symbol=' : texts.map(lambda x: x.count("="))
    })

def bag_of_words(texts, **args):
    vectorizer = CountVectorizer(**args)
    vectorizer.fit(texts)
    df = pd.DataFrame(vectorizer.transform(texts).toarray())
    df.columns = map(string.upper,vectorizer.get_feature_names())
    return df
    
def bag_of_words_tdidf(texts, **args):
    vectorizer = TfidfVectorizer(**args)
    vectorizer.fit(texts)
    df = pd.DataFrame(vectorizer.transform(texts).toarray())
    df.columns = map(string.upper,vectorizer.get_feature_names())
    return df

spellchecker = pyaspell.Aspell(("lang", "en"))
def spellcheck(text):
    text = str(text)
    print "spellchecking %s len: %d " % (text, len(text))
    correct_text = text
    for word in text.split():
        if not spellchecker.check(word) and word not in ["EQUATIONINCORRECT","EQUATIONINCORRECT"] and not word.replace(".","").isdigit():
            suggestion = spellchecker.suggest(word)
            if suggestion:
                correct_text = correct_text.replace(word,suggestion[0])
    return correct_text
    
def levenshtein(seq1, seq2):
    oneago = None
    thisrow = range(1, len(seq2) + 1) + [0]
    for x in xrange(len(seq1)):
        twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
        for y in xrange(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
    return thisrow[len(seq2) - 1]
    
def reduce_vocabulary(words,threshold=0.65):
    from scipy import spatial
    wv = joblib.load("/home/pawel/SpiderOak Hive/McGraw/engine/data_work/word2vec_vectors.job")
    print "reduce_vocabulary size (before indexing)", len(words)    
    words = sorted([w for w in words if w in wv.index])    
    print "reduce_vocabulary size (after indexing)", len(words)    
    groups = []
    taken = set()
    for ai, a in enumerate(words):
        if a in taken:
            continue
        group = [a]
        for b in words[ai:]:
            if b in taken:
                continue
            distance = 1 - spatial.distance.cosine(wv.ix[a],wv.ix[b])
            if distance > threshold:
                group += [b,]
                taken.add(b)
        groups.append(group)
        
    return groups
    
def replace_characters(text,list_of_chars,repl=" "):
    for char in list_of_chars:
        text = text.replace(char, repl)
    return text    
    
def reduce_vocabulary_dict(TEXTS, threshold=0.65):
    for key in TEXTS:  
        texts_ = TEXTS[key]
        vectorizer = CountVectorizer()
        vectorizer.fit(texts_)
        vocabulary = vectorizer.vocabulary_.keys()
        vocabulary = map(string.upper, vocabulary)
        print "reducing vocabulary", key
        print "vocabulary length", len(vocabulary)
        print "head", vocabulary[:30]
        vocabulary = sorted(vocabulary)
        reduced_vocab = reduce_vocabulary(vocabulary,threshold)
        for group in reduced_vocab:
            if len(group) > 1:
                texts_ = texts_.map(lambda x: replace_characters(x, group, group[0]))
        TEXTS[key] = texts_
    return TEXTS
                
def reduce_vocabulary_func(texts_, threshold=0.65):
    vectorizer = CountVectorizer()
    vectorizer.fit(texts_)
    vocabulary = vectorizer.vocabulary_.keys()
    vocabulary = map(string.upper, vocabulary)
    vocabulary = sorted(vocabulary)
    reduced_vocab = reduce_vocabulary(vocabulary,threshold)
    for group in reduced_vocab:
        if len(group) > 1:
            texts_ = texts_.map(lambda x: replace_characters(x, group, group[0]))
    return texts_
        
def clean_text(TEXTS):
    for f in [
           lambda x: re.sub("&[a-z]+?;","",x)
          ,lambda x: re.sub("<[^>]+?>","",x)
          ,lambda x: x.replace('&NBSP;',' ')
          ,string.upper
		  ,lambda x: x.replace('<P>',' ')
		  ,lambda x: x.replace('</P>',' ')
          ,lambda x: re.sub("([A-Z])'([A-Z])",'\g<1>\g<2>', x)
          # replaces , in digits (it should be .)
          ,lambda x: re.sub('([0-9]),([0-9])','\g<1>.\g<2>', x)
          # removes dots after a digit
          ,lambda x: re.sub('([0-9]). ','\g<1>', x)
          # keeps ratios
          ,lambda x: re.sub('([0-9]):([0-9])','\g<1>TO\g<2>', x)
          # removes dot at the end of the text
          ,lambda x: re.sub('.$','', x)
          # removes unnecessary characters
          ,lambda x: replace_characters(x, [",",'"',";","!"," . ",". "," .",'"',"' ","-"]," ")
          # removes dots near letters
          ,lambda x: re.sub('([A-Z])\.', '\g<1>', x)

          ]:
        TEXTS = func_over_dict(TEXTS,apply_map_func(f),parallel=True)
    return TEXTS
    
def pca_decomposition(bow,n_components):
    m = RandomizedPCA(n_components)
    return m.fit_transform(np.array(bow))
    
if __name__ == "__main__":
    """
    for word in list(BOW[BOW.keys()[0]].columns):
        word = str(word)
        correct = spellcheck(word)
        if word != correct:
            print word, correct
    """
    pass