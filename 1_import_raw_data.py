import os
import glob
from lib.utils import convert_xml_to_csv

from config import *

if __name__ == "__main__":
    # normal essays
    for xmlfile in glob.glob(os.path.join(MAIN_PATH,"engine","data_work","items_data","training","*.zip")):
        print "Converting", xmlfile
        filename = "_".join(os.path.split(xmlfile)[-1].split("_")[:2])
        itemid = os.path.splitext(filename)[0]
        outputfile = os.path.join(MAIN_PATH,"engine","data_work","items_data",itemid + ".csv")    
        convert_xml_to_csv(xmlfile, outputfile,COL_NAMES,READINGS_LIST)
        
    for xmlfile in glob.glob(os.path.join(MAIN_PATH,"engine","data_work","items_data","testing","*.zip")):
        print "Converting", xmlfile
        filename = "_".join(os.path.split(xmlfile)[-1].split("_")[:2])
        itemid = os.path.splitext(filename)[0]
        outputfile = os.path.join(MAIN_PATH,"engine","data_work","items_data",itemid + ".csv")    
        convert_xml_to_csv(xmlfile, outputfile,COL_NAMES,READINGS_LIST,append=True)
        
    # short essays
    for xmlfile in glob.glob(os.path.join(MAIN_PATH,"engine","data_work","items_data_se","training","*.zip")):
        print "Converting", xmlfile
        filename = "_".join(os.path.split(xmlfile)[-1].split("_")[:2])
        itemid = os.path.splitext(filename)[0]
        for data_point in ["A","B","C"]:
			outputfile = os.path.join(MAIN_PATH,"engine","data_work","items_data_se",itemid + "_data_point_%s.csv" % (data_point))    
			convert_xml_to_csv(xmlfile, outputfile,COL_NAMES,READINGS_LIST,data_point=data_point)

    for xmlfile in glob.glob(os.path.join(MAIN_PATH,"engine","data_work","items_data_se","testing","*.zip")):
        print "Converting", xmlfile
        filename = "_".join(os.path.split(xmlfile)[-1].split("_")[:2])
        itemid = os.path.splitext(filename)[0]
        for data_point in ["A","B","C"]:
			outputfile = os.path.join(MAIN_PATH,"engine","data_work","items_data_se",itemid + "_data_point_%s.csv" % (data_point))    
			convert_xml_to_csv(xmlfile, outputfile,COL_NAMES,READINGS_LIST,append=True,data_point=data_point)

    # special studies - gaming
    for xmlfile in glob.glob(os.path.join(MAIN_PATH,"engine","data_work","items_data_gaming","training","*.zip")):
        print "Converting", xmlfile
        filename = "_".join(os.path.split(xmlfile)[-1].split("_")[:2])
        itemid = os.path.splitext(filename)[0]
        outputfile = os.path.join(MAIN_PATH,"engine","data_work","gaming",itemid + ".csv")    
        convert_xml_to_csv(xmlfile, outputfile,COL_NAMES,READINGS_LIST)
        
    for xmlfile in glob.glob(os.path.join(MAIN_PATH,"engine","data_work","items_data_gaming","testing","*.zip")):
        print "Converting", xmlfile
        filename = "_".join(os.path.split(xmlfile)[-1].split("_")[:2])
        itemid = os.path.splitext(filename)[0]
        outputfile = os.path.join(MAIN_PATH,"engine","items_data_gaming_work","gaming",itemid + ".csv")    
        convert_xml_to_csv(xmlfile, outputfile,COL_NAMES,READINGS_LIST,append=True)

    # special studies - sample size    
    for xmlfile in glob.glob(os.path.join(MAIN_PATH,"engine","data_work","sample_size_1","training","*.xml")):
        print "Converting", xmlfile
        filename = "_".join(os.path.split(xmlfile)[-1].split("_")[:2])[2:]
        itemid = os.path.splitext(filename)[0]
        print filename
        outputfile = os.path.join(MAIN_PATH,"engine","data_work","sample_size_1",itemid + ".csv")    
        convert_xml_to_csv(xmlfile, outputfile,COL_NAMES,READINGS_LIST,data_point=None,filezipped=False)
        
    for samplesize in range(1,15+1):
        for xmlfile in glob.glob(os.path.join(MAIN_PATH,"engine","data_work","items_data_sample_size","testing","*.xml")):
            print "Converting", xmlfile
            filename = "_".join(os.path.split(xmlfile)[-1].split("_")[:1])[2:] + "_" + str(samplesize)
            itemid = os.path.splitext(filename)[0]
            print filename
            outputfile = os.path.join(MAIN_PATH,"engine","data_work","items_data_sample_size",itemid + ".csv")    
            convert_xml_to_csv(xmlfile, outputfile,COL_NAMES,READINGS_LIST,append=True,data_point=None,filezipped=False)

