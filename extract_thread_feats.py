# coding=utf-8
# python extract_thread_feats.py ../dataconversion/Viva_forum/samples/106long20threads 106long20threads.threadfeats.out
# python extract_thread_feats.py ../dataconversion/Viva_forum/samples/kankerthreads kankerthreads.threadfeats.out
# python extract_thread_feats.py ../dataconversion/GIST_FB/threads GIST_FB.threadfeats.out




import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import re
import math
import fileinput
from collections import defaultdict
import xml.etree.ElementTree as ET
#from sklearn import datasets
#from sklearn.feature_extraction import DictVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
#import matplotlib
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#from sklearn.datasets import make_blobs
#from sklearn.cluster import KMeans
#from sklearn.cluster import AgglomerativeClustering
#from sklearn.cluster import SpectralClustering
#import numpy
#from scipy.sparse import coo_matrix, hstack
from scipy.linalg import norm


rootdir = sys.argv[1]
outfile = sys.argv[2]

#postcounts = dict()

#postcountfilename = "../dataconversion/postcountperthread.reddit.txt"
#if re.match(".*Viva.*",rootdir):
#    postcountfilename = "../dataconversion/postcountperthread.viva.txt"

#sys.stderr.write("Read postcountperthread file\n")
#with open(postcountfilename,"r") as postcountfile:
#    for line in postcountfile:
#        parts = line.rstrip().split("\t")
#        filename = parts[0]
#        postcount = parts[1]
#        postcounts[filename] = postcount
        #print filename, postcount


#postcountfile.close()


#ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(4, 4), min_df=1)


def tokenize(t):
    text = t.lower()
    text = re.sub("\n"," ",text)
    text = re.sub('[^a-zèéeêëûüùôöòóœøîïíàáâäæãåA-Z0-9- \']', "", text)
    wrds = text.split()
    return wrds

def fast_cosine_sim(a, b):
    if len(b) < len(a):
        a, b = b, a
    up = 0
    for key, a_value in a.iteritems():
        b_value = b.get(key, 0)
        up += a_value * b_value
    if up == 0:
        return 0
    return up / norm(a.values()) / norm(b.values())

title_column = list()
threadid_column = list()
bodyoffirstpost_column = list()
postcount_column = list()
postcountcutoff_column = list()
category_column = list()
length_of_title_column = list()
length_of_opening_post_column = list()
avg_postlength_column = list()
questioncount_column = list()
avgcossimwiththread_column = list()
avgcossimwithprevious_column = list()

sys.stderr.write("Read files in "+rootdir+"\n")
for f in os.listdir(rootdir):
    if f.endswith("xml"):

        tree = ET.parse(rootdir+"/"+f)
        root = tree.getroot()
        print f


        for thread in root:
            threadid = thread.get('id')
            category = thread.find('category').text
            title = thread.find('title').text
            if title is None:
                title = ""

            termvectorforthread = dict()  # key is term, value is termcount for full thread
            termvectors = defaultdict(dict)  # key is postid, value is dict with term -> termcount for post
            cosinesimilaritiesthread = list()
            cosinesimilaritiesprevious = list()
            postcount_for_postid = dict()
            postid_for_postcount = dict()

            #sys.stderr.write(f+"\t"+title+"\n")


            threadid_column.append(threadid)
            title_column.append(title)
            length_of_title_column.append(len(title))
            category_column.append(category)

            sum_postlength = 0
            noofposts = 0
            for posts in thread.findall('posts'):
                firstpost = posts.findall('post')[0]
                bodyoffirstpost = firstpost.find('body').text
                bodyoffirstpost_column.append(bodyoffirstpost)
                if bodyoffirstpost is None:
                    length_of_opening_post_column.append(0)
                    questioncount_column.append(0)
                else:
                    length_of_opening_post_column.append(len(bodyoffirstpost))
                    questioncount_column.append(bodyoffirstpost.count('?'))
                postcount = 0
                for post in posts.findall('post'):
                    postid = post.get('id')
                    postcount_for_postid[postid] = postcount
                    postid_for_postcount[postcount] = postid
                    noofposts += 1
                    if postcount > 51:
                        continue

                    bodyofpost = post.find('body').text
                    #if threadid == "160543202978_10152800850187979":
                    #   print threadid, postid, bodyofpost

                    if bodyofpost is None:
                        bodyofpost = ""
                    words = tokenize(bodyofpost)
                    if bodyofpost is not None:
                        sum_postlength += len(bodyofpost)
                        for word in words:
                            #print word, nrofsyllables(word)
                            if word in termvectorforthread:  # dictionary over all posts
                               termvectorforthread[word] += 1
                            else:
                               termvectorforthread[word] = 1

                            if word in termvectors[postid].keys():  # dictionary per post
                               termvectors[postid][word] += 1
                            else:
                               termvectors[postid][word] = 1
                    postcount += 1

            noofposts_cutoff = noofposts
            if noofposts_cutoff > 50:
                noofposts_cutoff = 50
            postcount_column.append(noofposts)
            postcountcutoff_column.append(noofposts_cutoff)
            avg_postlength_column.append(sum_postlength/noofposts)
            #ngram_vectorizer.fit_transform(['jumpy fox'])
            #ngram_vectorizer.fit_transform([title])
            #print ngram_vectorizer.get_feature_names()


            for postid in termvectors.keys():
                termvectorforpost = termvectors[postid]
                for word in termvectorforthread:
                    if word not in termvectorforpost:
                        termvectorforpost[word] = 0
                termvectors[postid] = termvectorforpost

                cossimthread = fast_cosine_sim(termvectors[postid], termvectorforthread)
                cossimprevious = 0
                #print postid
                if postcount_for_postid[postid] > 1:
                    cossimprevious = fast_cosine_sim(termvectors[postid], termvectors[postid_for_postcount[postcount_for_postid[postid]-1]])

                cosinesimilaritiesthread.append(cossimthread)
                cosinesimilaritiesprevious.append(cossimprevious)

            avgcossimwiththread = np.average(cosinesimilaritiesthread)
            avgcossimwiththread_column.append(avgcossimwiththread)

            avgcossimwithprevious = np.average(cosinesimilaritiesprevious)
            avgcossimwithprevious_column.append(avgcossimwithprevious)

#sys.stderr.write("Compute matrices...\n")

#X = ngram_vectorizer.fit_transform(title_column)
#X = ngram_vectorizer.fit_transform(bodyoffirstpost_column)
#print ngram_vectorizer.fit(title_column).vocabulary_

#matrix = X.toarray()

nonsparse_dimensions = (postcount_column,postcountcutoff_column,length_of_title_column,length_of_opening_post_column,avg_postlength_column,questioncount_column,avgcossimwiththread_column,avgcossimwithprevious_column,category_column)

noofitems = len(title_column)
sys.stderr.write("No of items: "+str(noofitems)+"\n")
#noofdimensions_sparse = len(ngram_vectorizer.get_feature_names())
noofdimensions_nonsparse = len(nonsparse_dimensions)
#sys.stderr.write("No of dimensions (sparse): "+str(noofdimensions_sparse)+"\n")
sys.stderr.write("No of dimensions (nonsparse): "+str(noofdimensions_nonsparse)+"\n")


nonsparsematrix = [[0 for x in range(noofdimensions_nonsparse)] for y in range(noofitems)]

i=0
for title in title_column:
    instance_array=[]
    for column in nonsparse_dimensions:
        instance_array.append(column[i])
    nonsparsematrix[i] = instance_array
    i += 1

#print threadid_column
#print nonsparsematrix

threadfeatsfile = open(outfile,'w')

threadfeatsfile.write("threadid\tpostcount\tpostcount_cutoff\tlength_of_title\tlength_of_opening_post\tavg_postlength\tquestion_count\tavg_cossim_with_thread\tavg_cossim_with_previous\tcategory\n")
ci =0
for instance_array in nonsparsematrix:
    threadfeatsfile.write(threadid_column[ci])
    for value in instance_array:
        threadfeatsfile.write("\t"+str(value))
    threadfeatsfile.write("\n")
    ci += 1

threadfeatsfile.close()
#combimatrix = hstack([X,nonsparsematrix]).toarray()

#http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.sparse.coo_matrix.html

#X._shape = (noofitems,noofdimensions+1)
#i=0
#j=noofdimensions-1

#for value in postcount_column:
#    sys.stderr.write("i:"+str(i)+" j:"+str(j)+"\n")
#    X[i,j] = value
#    i += 1

#print X.toarray()

#sys.stderr.write("Cluster threads...\n")

#for title in title_column:
#    sys.stderr.write(title+"\n")

#nclusters_values = (2,4,8)
#nclusters_values = [4]

#for nclusters in nclusters_values:
    #kmeans = KMeans(n_clusters=nclusters)
    #spectral = SpectralClustering(n_clusters=nclusters)
    #agglom = AgglomerativeClustering(n_clusters=nclusters)
    #print kmeans.fit(X)

    #kmeans_predictions = kmeans.fit_predict(X)
    #kmeans_predictions_nonsparse = kmeans.fit_predict(nonsparsematrix)
    #kmeans_predictions_combi = kmeans.fit_predict(combimatrix)
    #spectral_predictions = spectral.fit_predict(X)

    #print "n_clusters:", nclusters, "\tinertia:", kmeans.inertia_
    #print "n_clusters:", nclusters, "\tn_leaves:", agglom.n_leaves_

    #predictions = kmeans_predictions
    #predictions = kmeans_predictions_nonsparse
    #predictions = kmeans_predictions_combi
    #predictions = spectral_predictions
    #sys.stderr.write("No of predictions:"+str(len(predictions))+"\n")
    #clusters = dict()
    #j=0
    #for title in title_column:
        #sys.stderr.write(title+"\t")
        #sys.stderr.write(str(predictions[j])+"\n")
        #cluster_id = predictions[j]
        #cluster = list()
        #if clusters.has_key(cluster_id):
       #     cluster = clusters[cluster_id]
      #  cluster.append(title)
      #  clusters[cluster_id] = cluster
      #  j += 1

    #print "\n\n>", nclusters, "clusters\n\n"
    #for cluster_id in clusters:
       # cluster = clusters[cluster_id]
       # clustersize = len(cluster)
       # print "\n--------\nCLUSTER", cluster_id, "(", clustersize,"threads)\n--------\n"
       # i=0
       # for title in cluster:
       #     featvalues = ""
      #      for column in nonsparse_dimensions:
      #          featvalues += ","+str(column[i])
      #      print "\t-", title,"(",featvalues,")"
      #      i += 1