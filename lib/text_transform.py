import re
#from stemming import porter
from string import lower, printable, upper

STOPWORDS = ['a','able','about','across','after','all','almost','also','am','among',
             'an','and','any','are','as','at','be','because','been','but','by','can',
             'cannot','could','dear','did','do','does','either','else','ever','every',
             'for','from','get','got','had','has','have','he','her','hers','him','his',
             'how','however','i','if','in','into','is','it','its','just','least','let',
             'like','likely','may','me','might','most','must','my','neither','no','nor',
             'not','of','off','often','on','only','or','other','our','own','rather','said',
             'say','says','she','should','since','so','some','than','that','the','their',
             'them','then','there','these','they','this','tis','to','too','twas','us',
             'wants','was','we','were','what','when','where','which','while','who',
             'whom','why','will','with','would','yet','you','your']
STOPWORDS += map(upper,STOPWORDS)

# basic text transformations
def remove_all_non_printable(text):
    return "".join([k for k in text if k in printable])

def remove_all_non_characters(text):
    return re.sub("[^a-zA-Z\s]"," ",text)

def remove_non_number(text):
    return re.sub("[^0-9\s]"," ",text).strip()

def remove_multispaces(text):
    return re.sub("[\s]+"," ",text)

def remove_spaces(text):
    return re.sub("[\s]","",text)

def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in STOPWORDS])

def replace_consecutive_letters(text):
    letters = [k for k in text if k not in [" ",]]
    for letter in letters:
        text = re.sub("[%s]+" % (letter,), letter, text)
    return text
    
def replace_characters(text,list_of_chars,repl=" "):
    for char in list_of_chars:
        text = text.replace(char, repl)
    return text

def apply_func_to_words(text, func):
    return " ".join([func(w) for w in text.split()])

def stem(text, func):
    return apply_func_to_words(text, func)
    
def apply_text_functions(text, funcs=[remove_all_non_printable,
                                      remove_all_non_characters,
                                      remove_multispaces,
                                      remove_stopwords]):
    for func in funcs:
        text = func(text)
    return text

def hashingtrick(document,size,salt=""):
    bow=[0]*size
    for word in document.split():
        word += salt
        h=word.__hash__()
        if h < 0:
            bow[abs(h) % size] -= 1
        else:
            bow[abs(h) % size] += 1
    return bow