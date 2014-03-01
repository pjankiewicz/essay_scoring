import glob

from lib.pipeline import spellcheck
from xml.dom import minidom
import os

for xmlpath in glob.glob("/home/pawel/SpiderOak Hive/McGraw/engine/data_work/items_data/training/*.xml") + \
           glob.glob("/home/pawel/SpiderOak Hive/McGraw/engine/data_work/items_data/testing/*.xml"):
    
    print filename    
    folder, filename = os.path.split(xmlpath)
    xml_text = open(xmlpath).read()

    dom = minidom.parseString(xml_text)
    responses = dom.getElementsByTagName("Item_Response")
    for response in responses:
        text = response.firstChild.nodeValue
        text_spellchecked = spellcheck(text)
        response.firstChild.replaceWholeText(text_spellchecked)

    filename_spellchecked = filename[:-4] + "_spellcheck.xml"
    out = open("/home/pawel/SpiderOak Hive/McGraw/engine/spellcheck/" + filename_spellchecked,"w")
    out.write(dom.toxml())
    out.close()


"""
for param in params:
    name=param.getAttribute("name")
    if name in data:
        for item in param.getElementsByTagName("*"): # You may change to "Result" or "Value" only
            item.firstChild.replaceWholeText(data[name])
"""


#write to file
open("output.xml","wb").write(dom.toxml())