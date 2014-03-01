import re
import string

def replaces_text_numbers(text):
    for p,r in [("ONE","1"),("TWO","2"),("THREE","3"),("FOUR","4"),
                ("FIVE","5"),("SIX","6"),("SEVEN","7"),("EIGHT","8"),
                ("NINE","1"),("ZERO","0")]:
        text = text.replace(" %s " % (p)," %s " % (r))
    return text
    
def evaluate_expressions(text):
    for p in re.findall("([0-9+-/*]+)",text):
        try:
            text = text.replace(p,str(eval(p)))
        except:
            pass
    return text

def check_equalities(text):
    correct = 0
    incorrect = 0
    for l,r in re.findall("([0-9\.]+)=([0-9\.]+)",text):
        if l.endswith("."): l = l[:-1]
        if r.endswith("."): r = r[:-1]
        if float(l)==float(r):
            text = text.replace("%s=%s" % (l,r), " EQUATIONCORRECT ")
        else:
            text = text.replace("%s=%s" % (l,r), " EQUATIONINCORRECT ")
            
    for l,r in re.findall("([0-9\.]+)!=([0-9\.]+)",text):
        if l.endswith("."): l = l[:-1]
        if r.endswith("."): r = r[:-1]
        if float(l)==float(r):
            text = text.replace("%s=%s" % (l,r), " EQUATIONINCORRECT ")
        else:
            text = text.replace("%s=%s" % (l,r), " EQUATIONCORRECT ")
    return text
            
    
def simplify_math(text):
    text = str(text).upper()
    text = text.replace("\\",'/')
    text = replaces_text_numbers(text)
    text = re.sub('([\s])+', ' ', text)
    # replaces 1.8C with 1.8
    
    # removes spaces
    for p, r in [
         ("([0-9])[cC]","\g<1>")
        ,("([0-9]+)[ ]{0,}\+[ ]{0,}([0-9]+)","\g<1>+\g<2>")
        ,("([0-9]+)[\s]?=[\s]?([0-9]+)","\g<1>=\g<2>")
        ,("([0-9]+)[\s]?-[\s]?([0-9]+)","\g<1>-\g<2>")
        ,("([0-9]+)[\s]?\*[\s]?([0-9]+)","\g<1>*\g<2>")
        ,("([0-9]+)[\s]?/[\s]?([0-9]+)","\g<1>/\g<2>")
        
        # 
        ,("([0-9]+) IS * BY ([0-9]+)","\g<1>*\g<2>")
        ,("([0-9]+) IS MULTIPLIED BY ([0-9]+)","\g<1>*\g<2>")
        ,("([0-9]+) MULTIPLIED ([0-9]+)","\g<1>*\g<2>")
        ,("([0-9]+) TIMES ([0-9]+)","\g<1>*\g<2>")
        ,("([0-9]+) X ([0-9]+)","\g<1>*\g<2>")
        ,("([0-9]+) MULTIPLY BY ([0-9]+)","\g<1>*\g<2>")
        ,("([0-9]+) PLUS ([0-9]+)","\g<1>+\g<2>")
        ,("([0-9]+) MINUS ([0-9]+)","\g<1>-\g<2>")
        ,("([0-9]+) OVER ([0-9]+)","\g<1>/\g<2>")
        ,("([0-9]+) DIVIDED BY ([0-9]+)","\g<1>/\g<2>")
        
        # equality    
        ,("([0-9]+) IS ([0-9]+)","\g<1>=\g<2>")
        ,("([0-9]+) EQUALS ([0-9]+)","\g<1>=\g<2>")
        ,("([0-9]+) EQUALS TO ([0-9]+)","\g<1>=\g<2>")
        ,("([0-9]+) EQUALS THE SAME AS THE FRACTION ([0-9]+)","\g<1>=\g<2>")
        ,("([0-9]+) IS ([0-9]+)","\g<1>=\g<2>")
        ,("([0-9]+) IT IS ([0-9]+)","\g<1>=\g<2>")
        ,("([0-9]+) IS JUST ([0-9]+)","\g<1>=\g<2>")
        ,("([0-9]+) IS EQUAL TO ([0-9]+)","\g<1>=\g<2>")
        ,("([0-9]+) IS EQUIVALENT TO ([0-9]+)","\g<1>=\g<2>")
        ,("([0-9]+) IS = TO ([0-9]+)","\g<1>=\g<2>")
        ,("([0-9]+) CAN ALSO TURN TO ([0-9]+)","\g<1>=\g<2>")
        ,("([0-9]+) CAN TURN TO ([0-9]+)","\g<1>=\g<2>")
        ,("([0-9]+) IS A DECIMAL FOR ([0-9]+)","\g<1>=\g<2>")
        ,("([0-9]+) IN DECIMAL FORM IS ([0-9]+)","\g<1>=\g<2>")
        ,("([0-9]+) IS THE DECIMAL FOR ([0-9]+)","\g<1>=\g<2>")
        ,("([0-9]+) IS THE SAME AS ([0-9]+)","\g<1>=\g<2>")
        ,("([0-9]+) IS THE SAME AS THE FRACTION ([0-9]+)","\g<1>=\g<2>")
        ,("([0-9]+) IS THE ANSWER TO ([0-9]+)","\g<1>=\g<2>")
        
        # inequality    
        ,("([0-9]+) IS NOT THE SAME AS ([0-9]+)","\g<1>!=\g<2>")
        ,("([0-9]+) IS A FRACTION FOR ([0-9]+)","\g<1>!=\g<2>")
        ,("([0-9]+) IS NOT A FORM ([0-9]+)","\g<1>!=\g<2>")
        ,("([0-9]+) IS NOT EQUIVALENT TO ([0-9]+)","\g<1>!=\g<2>")
        
        # < > 
        ,("([0-9]+) IS BIGGER THAN ([0-9]+)","\g<1>>\g<2>")
        ,("([0-9]+) IS BIGER THAN ([0-9]+)","\g<1>>\g<2>")
        
        ]:
        text = re.sub(p, r, text)
    
    text = evaluate_expressions(text)
    text = check_equalities(text)
    
    return text

if __name__ == "__main__":
    print simplify_math("2 PLUS 2 EQUALS 4")
    print simplify_math("2/2 PLUS 1 EQUALS 2")
    print simplify_math("2 TIMES 2 PLUS 2 EQUALS 7")
    