# coding=utf-8
# python get_char_ngrams.py ../dataconversion/NYtimes/threads nytimes.charngrams.tsv

# output: csv file with group id in first column, the label in second column and features in all following columns

import operator

import os
import sys
import re
#import time
features = dict() # as reference, to keep track of all unique ngrams


import xml.etree.ElementTree as ET
# from xml.dom import minidom

rootdir = sys.argv[1]
featfilename = sys.argv[2]

def get_char_ngrams (t):
    global features
    ngrams = dict()

    for i in range(0,len(t)-3):
        ngram = t[i:i+3]
        if ngram in ngrams:
            ngrams[ngram] +=1
        else:
            ngrams[ngram] = 1
        if ngram in features:
            features[ngram] += 1
        else:
            features[ngram] = 1
    #print(ngrams)
    return ngrams

def tokenize(t):
    text = t.lower()
    text = re.sub("\n"," ",text)
    text = re.sub('[^a-zèéeêëûüùôöòóœøîïíàáâäæãåA-Z0-9- \']', "", text)
    wrds = text.split()
    return wrds

caps = "([A-Z])"
prefixes = "(Dhr|Mevr|Dr|Drs|Mr|Ir|Ing)[.]"
suffixes = "(BV|MA|MSc|BSc|BA)"
starters = "(Dhr|Mevr|Dr|Drs|Mr|Ir|Ing)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|nl)"


print ("Read files in ",rootdir,"\n")


ngram_dict_per_post = dict() # key is threadid, postid
postvotes = dict() # key is threadid, postid

threadid_to_groupid = dict()
groupid = 0

for f in os.listdir(rootdir):
    if f.endswith("xml"):
        #print (time.clock(), "\t", f)
        print (f)

        postids = list()

        tree = ET.parse(rootdir+"/"+f)
        root = tree.getroot()

        for thread in root:
            threadid = thread.get('id')
            if not threadid in threadid_to_groupid:
                groupid += 1
                threadid_to_groupid[threadid] = groupid

            for posts in thread.findall('posts'):
                noofposts = len(posts.findall('post'))
                if noofposts > 50:
                    noofposts = 50
                postcount = 0


                for post in posts.findall('post'):
                    postcount += 1
                    postid = post.get('id')
                    #print (threadid,postid)


                    postids.append(postid)
                    #postids_dict[(threadid,postid)] = postid
                    #threadids[(threadid,postid)] = threadid

                    selected = post.find('selected').text
                    # in the NYtimes data, the postvotes are the editor picks ('selected')
                    #  - they are part of the original data and included in the XML
                    if selected is not None:
                        postvotes[(threadid,postid)] = selected


                    body = post.find('body').text
                    if postcount > 51:
                        continue
                    elif postid=="0":
                        continue
                    elif body is None:
                        body = ""

                    char_ngrams = get_char_ngrams(body)
                    ngram_dict_per_post[(threadid,postid)] = char_ngrams




print ("dimensionality before feature selection:",len(features))
features_selected = dict()
features_sorted_by_frequency = sorted(features.items(), key=operator.itemgetter(1),reverse=True)
#for feat in features:
#    if features[feat] >= 10:
#        features_selected[feat] = features[feat]
k = 0
for (feat,freq) in features_sorted_by_frequency:
    k += 1
    features_selected[feat] = freq
    if k >= 1000:
        break
print ("dimensionality after feature selection:",len(features_selected))
print (features_selected)


features_sorted = sorted(features_selected.items(), key=operator.itemgetter(0))
featfile = open(featfilename,'w')
print ("Print feature file to ",featfilename,"\n")

for (threadid,postid) in ngram_dict_per_post:
    #print (threadid,postid)
    char_ngrams_for_this_post = ngram_dict_per_post[(threadid,postid)]
    feat_vector = [threadid,postid]
    # First, add group id, in first column (for pairwise preference learning, we need grouped data. The threads are the groups)

    for (feat,count) in features_sorted:
        if feat in char_ngrams_for_this_post:
            feat_vector.append(char_ngrams_for_this_post[feat])
        else:
            feat_vector.append(0)
    feat_vector.append(postvotes[(threadid, postid)])
    # Lastly, add the label
    #print (len(feat_vector),threadid,postid,feat_vector)
    vector_print = "\t".join(str(f) for f in feat_vector)
    featfile.write(vector_print+"\n")


featfile.close()