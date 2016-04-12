# coding=utf-8
# python extract_post_feats.py ../dataconversion/Viva_forum/samples/106long20threads 106long20threads.postfeats.norm.out
# python extract_post_feats.py ../dataconversion/Viva_forum/samples/20randomthreads 20randomthreads.postfeats.norm.out
# python extract_post_feats.py ../dataconversion/GIST_FB/threads/ gistfb.postfeats.norm.out
# python extract_post_feats.py ../dataconversion/GIST_FB/smallsample/ gistfb.postfeats.norm.smallsample.out

# Feature extraction.
# + (a) position in the thread (absolute, relative)
# + (b) popularity (#responses)
# + (c) representativeness for the thread (cosine similarity for tf-idf weighted vector representations of post and thread/title)
# + (d) readability (wordcount, uniquewordcount,type-token ratio, relative punctuation count, average word length, avg sent length
# - Flesch(-Douma) statistic for Readability Ease))
# + (e) prominence of author (# relative posts in thread)


import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import re
import math
import string
import operator
import functools
import numpy
from scipy.linalg import norm
import time

import fileinput

from collections import defaultdict
import xml.etree.ElementTree as ET
# from xml.dom import minidom

rootdir = sys.argv[1]
featfilename = sys.argv[2]
annotationsfile = "../annotation/annotations/selected_posts.txt"




focususers = ("F.","Niree Bakker","Judith","Lian Bouten")

print time.clock(), "\t", "read annotations"

########### FROM ANALYZE_ANNOTATIONS.PY ###########
threads_per_user = dict()
age_per_user = dict()
freq_per_user = dict()
sona_per_user = dict()
email_per_user = dict()
selected_per_thread_and_user = dict()
utility_scores_per_thread = dict()
nrselecteds_per_thread = dict()
nrselecteds_per_user = dict()
with open(annotationsfile,'r') as annotations:
    for line in annotations:
        columns = line.split("\t")
        #Mon Jan 25 12:09:12 2016	Nikki de Groot	19	female	not	nikkidegroot24@gmail.com	97963
        name = columns[1]
        email = columns[5]
        if len(columns) > 7:
            # Mon Jan 25 12:14:31 2016	Nikki de Groot	19	female	not	nikkidegroot24@gmail.com	97963	103511	 2 3 6 7 8 9 10 12 13 22 24 28	2	4

            threadid = columns[7]
            selected = columns[8]

            if (re.match(".*[a-zA-Z0-9].*",name) and not re.match(r'(?i).*(test|suzan).*',name)) \
                    and (not re.match(r'voorbeeld',threadid) and (not threadid == "239355") and (not threadid == "250167")):
                #print name
                threadsforuser = dict()
                if threads_per_user.has_key(name):
                    threadsforuser = threads_per_user[name]
                if not threadsforuser.has_key(threadid):
                    threadsforuser[threadid] = 1
                    #ann.write(line)
                    nrselecteds = list()
                    threads_per_user[name] = threadsforuser
                    selected_def = dict()
                    postids = selected.split(" ")
                    removeatpos = dict()
                    pos=0
                    for postid in postids:
                        if "-" in postid:
                            removeid = re.sub("-","",postid)
                            removeatpos[removeid] = pos
                        pos += 1
                    pos = 0
                    for postid in postids:
                        if re.match("[0-9]+",postid):
                            if postid in removeatpos:
                                if pos > removeatpos[postid]:
                                    selected_def[postid] =1
                            else :
                                selected_def[postid]=1

                    selected_per_thread_and_user[(threadid,name)] = selected_def
                    nrselected = len(selected_def)
                    nrselecteds.append(nrselected)
                    nrselecteds_per_thread[threadid] = nrselecteds

                    nrselecteds_user = list()
                    if nrselecteds_per_user.has_key(name):
                        nrselecteds_user = nrselecteds_per_user[name]
                    nrselecteds_user.append(nrselected)
                    nrselecteds_per_user[name] = nrselecteds_user
                    #print selected, selected_def
                #else:
                #    sys.stderr.write("Warning: user "+email+" got thread "+threadid+" twice!\n")


print time.clock(), "\t", "collect postvotes per thread"

postvotes_per_thread = dict()
#out.write("\nthreadid\tuser\t# of selected posts\n")
for (threadid,user) in selected_per_thread_and_user.keys():
    selected = selected_per_thread_and_user[(threadid,user)]
    #print threadid, user, selected_per_thread_and_user[(threadid,user)], len(selected.keys())
    #out.write(threadid+"\t"+user+"\t"+str(len(selected.keys()))+"\n")

    postvotes = dict()
    if threadid in postvotes_per_thread.keys():
        postvotes = postvotes_per_thread[threadid]
    for postid in selected:
        if postid in postvotes.keys():
            postvotes[postid] += 1
        else:
            postvotes[postid] =1
    postvotes_per_thread[threadid] = postvotes

###########



def replace_quote(postcontent):
    adapted = ""
    blocks = re.split("\n\n",postcontent)
    #print blocks
    # first, find the block with the quote:
    bi=0
    bc = len(blocks)
    quoteblocki = 4
    while bi < bc:
        if " schreef op " in blocks[bi]:
            #print blocks[bi]
            quoteblocki = bi
            break

        # print until the quote:
        if not re.match('^>',blocks[bi]):
            adapted += blocks[bi]+"<br>\n"

        bi += 1
    blocks[quoteblocki] = re.sub("^>","",blocks[quoteblocki])
    blocks[quoteblocki] = re.sub("\(http://.*\):","",blocks[quoteblocki])
    #quote = blocks[quoteblocki]+blocks[quoteblocki+1]+"<br>\n"+blocks[quoteblocki+2]+"<br>\n"+blocks[quoteblocki+3]+"<br>\n"
    quote = blocks[quoteblocki]+"<br>\n"
    if len(blocks) > quoteblocki+1:
        quote += blocks[quoteblocki+1]+"<br>\n"

    #adapted += "<div style='background-color:rgb(240,240,240);padding-left:4em;padding-right:4em'>"+quote+"</div><br>\n"

    bi = quoteblocki+1
    while bi < bc:
        adapted += blocks[bi]+"<br>\n"
        bi += 1

    return adapted

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

def split_into_sentences(text):
    # adapted from http://stackoverflow.com/questions/4576077/python-split-text-on-sentences
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = re.sub("([\.\?!]+\)?)","\\1<stop>",text)
    if "<stop>" not in text:
        text += "<stop>"
    text = text.replace("<prd>",".")
    text = re.sub('  +',' ',text)
    sents = text.split("<stop>")
    sents = sents[:-1]
    sents = [s.strip() for s in sents]
    return sents

def count_punctuation(t):
    punctuation = string.punctuation
    punctuation_count = len(filter(functools.partial(operator.contains, punctuation), t))
    textlength = len(t)
    relpc = 0
    if textlength>0:
        relpc = float(punctuation_count)/float(textlength)
    return relpc

def nrofsyllables(w):
    count = 0
    vowels = 'aeiouy'
    w = w.lower().strip(".:;?!")
    if w[0] in vowels:
        count +=1
    for index in range(1,len(w)):
        if w[index] in vowels and w[index-1] not in vowels:
            count +=1
    if w.endswith('e'):
        count -= 1
    if w.endswith('le'):
        count+=1
    if count == 0:
        count +=1
    return count

def readability(wl,sl):
    # English: Flesch-Kincaid: 206,835 – (0.85 x nr of syllables per 100 words) – (1.015 x avg nr of words per sentence)
    # Dutch: Flesch-Douma: 206.84 - (0.77 * nr of syllables per 100 words) - (0.33 * ( words / Sentences))
    flesch = 206.84 - 0.77 * 100* wl - 0.33 * sl
    return flesch

def printTopNPosts(d, n):
    i = 0
    for pid in sorted(d, key=d.get, reverse=True):
        i += 1
        if i > n:
            break
        else:
            print pid, d[pid]
            # print postid, bodies[pid], d[pid]


def getTopNAuthors(d, n, minnrofposts):
    topauthors = list()
    i = 0
    print "    Top" + str(n) + " authors:"
    for aut in sorted(d, key=d.get, reverse=True):
        i += 1
        print "\t" + aut, d[aut]
        if i > n or d[aut] < minnrofposts:
            break
        else:
            topauthors.extend([aut])
    return topauthors


def normalizeVector(termvector):
    sumofsquares = 0
    for component in termvector.values():
        sumofsquares = sumofsquares + component * component
    vectorlength = math.sqrt(sumofsquares)
    for term in termvector.keys():
        normvalue = termvector[term] / vectorlength
        termvector[term] = normvalue
    return termvector


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


columns = dict() # key is feature name, value is dict with key (threadid,postid) and value the feature value

def addvaluestocolumnsoverallthreads(dictionary,feature):
    global columns
    columndict = dict()
    if feature in columns: # if this is not the first thread, we already have a columndict for this feature
        columndict = columns[feature] # key is (threadid,postid) and value the feature value
    for (threadid,postid) in dictionary:
        value = dictionary[(threadid,postid)]
        columndict[(threadid,postid)] = value
    #print feature, columndict
    columns[feature] = columndict

def standardize_values(columndict,feature):
    values = list()
    for (threadid,postid) in columndict:
        values.append(columndict[(threadid,postid)])
    mean = numpy.mean(values)
    stdev = numpy.std(values)
    normdict = dict() # key is (threadid,postid) and value the normalized feature value
    for (threadid,postid) in columndict:
        value = columndict[(threadid,postid)]
        normvalue = 0.0
        if stdev == 0.0:
            stdev = 0.000000000001
            print "stdev is 0! ", feature, value, mean, stdev
        #if value != 0:
        normvalue = (float(value)-float(mean))/float(stdev)
        normdict[(threadid,postid)] = normvalue
#        if feature == "noofupvotes":
#           print threadid,postid, feature, float(value), mean, stdev, normvalue, len(columndict)

    return normdict

sys.stderr.write("Read files in "+rootdir+"\n")

#for focususer in focususers:

#featfile = open("postfeats."+focususer+".out",'w')

openingpost_for_thread = dict() # key is threadid, value is id of opening post
postids_dict = dict() # key is (threadid,postid), value is postid. Needed for pasting the columns at the end
threadids = dict() # key is (threadid,postid), value is threadid. Needed for pasting the columns at the end
upvotecounts = dict()  # key is (threadid,postid), value is # of upvotes
responsecounts = dict()  # key is (threadid,postid), value is # of replies
cosinesimilaritiesthread = dict()  # key is (threadid,postid), value is cossim with term vector for complete thread
cosinesimilaritiestitle = dict()  # key is (threadid,postid), value is cossim with term vector for title
uniquewordcounts = dict()  # key is (threadid,postid), value is unique word count in post
wordcounts = dict() # key is (threadid,postid), value is word count in post
typetokenratios = dict()  # key is (threadid,postid), value is type-token ratio in post
abspositions = dict() # key is (threadid,postid), value is absolute position in thread
relpositions = dict()  # key is (threadid,postid), value is relative position in thread
relauthorcountsinthreadforpost = dict()  # key is (threadid,postid), value is relative number of posts by author in this thread
relpunctcounts = dict() # key is (threadid,postid), value is relative punctuation count in post
avgwordlengths = dict() # key is (threadid,postid), value is average word length (nr of characters)
avgnrsyllablesinwords = dict() # key is (threadid,postid), value is average word length (nr of syllables)
avgsentlengths = dict() # key is (threadid,postid), value is average word length (nr of words)
readabilities = dict() # key is (threadid,postid), value is readability
bodies = dict()  # key is (threadid,postid), value is content of post

#print time.clock(), "\t", "go through files"

for f in os.listdir(rootdir):
    if f.endswith("xml"):
        print time.clock(), "\t", f

        postids = list()
        termvectors = dict()  # key is postid, value is dict with term -> termcount for post
        termvectorforthread = dict()  # key is term, value is termcount for full thread
        termvectorfortitle = dict()  # key is term, value is termcount for title
        authorcountsinthread = dict()  # key is authorid, value is number of posts by author in this thread

        tree = ET.parse(rootdir+"/"+f)
        root = tree.getroot()

        for thread in root:
            threadid = thread.get('id')
            category = thread.find('category').text

            title = thread.find('title').text
            if title is None:
                # no title, then use complete opening post instead
                for posts in thread.findall('posts'):
                    title = posts.findall('post')[0].find('body').text

            if title is None:
                title = ""

            #print threadid,title

            titlewords = tokenize(title)
            for tw in titlewords:
                if tw in termvectorfortitle:
                    termvectorfortitle[tw] += 1
                else:
                    termvectorfortitle[tw] = 1
        # first go through the thread to find all authors
        for posts in thread.findall('posts'):
            for post in posts.findall('post'):
                author = post.find('author').text
                if author in authorcountsinthread:
                    authorcountsinthread[author] += 1
                else:
                    authorcountsinthread[author] =1

        for posts in thread.findall('posts'):
            noofposts = len(posts.findall('post'))
            if noofposts > 50:
                noofposts = 50
            postcount = 0

            #print time.clock(), "\t", "extract feats from each post"

            for post in posts.findall('post'):
                postcount += 1
                postid = post.get('id')

                if postcount == 1:
                    openingpost_for_thread[threadid] = postid
                    #print "opening post:",postid
                #print postid, postcount
                if 1 < postcount <= 51:
                    # don't include opening post in feature set
                    # and include at most 50 responses
                    postids.append(postid)
                    postids_dict[(threadid,postid)] = postid
                    threadids[(threadid,postid)] = threadid
                    parentid = post.find('parentid').text

                    if parentid != openingpost_for_thread[threadid]:
                        # do not save responses for openingpost because openingpost will not be in feature file
                        # (and disturbs the column for standardization)
                        if (threadid,parentid) in responsecounts:
                            responsecounts[(threadid,parentid)] += 1
                        else:
                            responsecounts[(threadid,parentid)] = 1
                    #else:
                        #print "don't add responsecounts because opening post:", parentid

                    upvotes = 0
                    upvotesitem = post.find('upvotes')
                    if upvotesitem is None:
                        upvotes = 0
                    else :
                        upvotes = upvotesitem.text
                    if upvotes is None:
                        upvotes = 0
                    #print threadid,postid,upvotes, responsecounts[(threadid,parentid)]
                    upvotecounts[(threadid,postid)] = int(upvotes)


                    body = post.find('body').text
                    if postcount > 51:
                        continue
                    elif postid=="0":
                        continue
                    elif body is None:
                        body = ""
                    author = post.find('author').text
                    relauthorcountsinthreadforpost[(threadid,postid)] = float(authorcountsinthread[author])/float(noofposts)
                    #print threadid, postid, author, authorcountsinthread[author]

                    if " schreef op " in body:
                        body = replace_quote(body)
                        #print threadid, postid, body

                    if "smileys" in body:
                        body = re.sub(r'\((http://forum.viva.nl/global/(www/)?smileys/.*.gif)\)','',body)

                    if "http" in body:
                        body = re.sub(r'http://[^ ]+','',body)

                    bodies[(threadid,postid)] = body

                    words = tokenize(body)
                    wc = len(words)

                    sentences = split_into_sentences(body)
                    sentlengths = list()

                    for s in sentences:
                        sentwords = tokenize(s)
                        nrofwordsinsent = len(sentwords)
                        #print s, nrofwordsinsent
                        sentlengths.append(nrofwordsinsent)
                    if len(sentences) > 0:
                        avgsentlength = numpy.mean(sentlengths)
                        avgsentlengths[(threadid,postid)] = avgsentlength
                    else:
                        avgsentlengths[(threadid,postid)] = 0
                    relpunctcount = count_punctuation(body)
                    relpunctcounts[(threadid,postid)] = relpunctcount
                    #print body, punctcount
                    wordcounts[(threadid,postid)] = wc
                    uniquewords = dict()
                    wordlengths = list()
                    nrofsyllablesinwords = list()
                    for word in words:
                        #print word, nrofsyllables(word)
                        nrofsyllablesinwords.append(nrofsyllables(word))
                        wordlengths.append(len(word))
                        uniquewords[word] = 1
                        if word in termvectorforthread:  # dictionary over all posts
                           termvectorforthread[word] += 1
                        else:
                           termvectorforthread[word] = 1

                        worddict = dict()
                        if postid in termvectors:
                            worddict = termvectors[postid]
                        if word in worddict:
                            worddict[word] += 1
                        else:
                            worddict[word] = 1
                        termvectors[postid] = worddict

                    uniquewordcount = len(uniquewords.keys())
                    uniquewordcounts[(threadid,postid)] = uniquewordcount
                    readabilities[(threadid,postid)] = 0

                    if wc > 0:
                        avgwordlength = numpy.mean(wordlengths)
                        #avgnrsyllablesinword = numpy.mean(nrofsyllablesinwords)
                        avgwordlengths[(threadid,postid)] = avgwordlength
                        #avgnrsyllablesinwords[(threadid,postid)] = avgnrsyllablesinword
                        #readabilities[(threadid,postid)] = readability(avgnrsyllablesinword,avgsentlength)
                    else:
                        avgwordlengths[(threadid,postid)] = 0

                    #print threadid, postid, wc, avgnrsyllablesinword, avgsentlength, readability(avgnrsyllablesinword,avgsentlength)

                    typetokenratio = 0
                    if wordcounts[(threadid,postid)] > 0:
                        typetokenratio = float(uniquewordcount) / float(wordcounts[(threadid,postid)])
                    typetokenratios[(threadid,postid)] = typetokenratio

                    relposition = float(postcount)/float(noofposts)
                    #relposition = float(postid)/float(noofposts)
                    relpositions[(threadid,postid)] = relposition
                    abspositions[(threadid,postid)] = postcount

                    #abspositions[(threadid,postid)] = postid


        #print time.clock(), "\t", "fill term vectors"

        #print wordcounts
        # add zeroes for titleterms that are not in the thread vector
        for titleword in termvectorfortitle:
            if titleword not in termvectorforthread:
                termvectorforthread[titleword] = 0

        # add zeroes for terms that are not in the title vector:
        for word in termvectorforthread:
            if word not in termvectorfortitle:
                termvectorfortitle[word] = 0

        # add zeroes for terms that are not in the post vector:
        for postid in termvectors:
            worddictforpost = termvectors[postid]
            for word in termvectorforthread:
                if word not in worddictforpost:
                    worddictforpost[word] = 0
            termvectors[postid] = worddictforpost


        #    for term in termvectorforthread:
        #        print(postid,term,termvectors[postid][term])
            cossimthread = fast_cosine_sim(termvectors[postid], termvectorforthread)
            cossimtitle = fast_cosine_sim(termvectors[postid], termvectorfortitle)
            cosinesimilaritiesthread[(threadid,postid)] = cossimthread
            cosinesimilaritiestitle[(threadid,postid)] = cossimtitle

            #print postid, cossimthread

        postvotes_for_this_thread = dict()
        if threadid in postvotes_per_thread:
            postvotes_for_this_thread = postvotes_per_thread[threadid]

        # no upvotes in Viva data

        #if (threadid,focususer) in selected_per_thread_and_user:
        #    selected_by_focususer = selected_per_thread_and_user[(threadid,focususer)]

        #print time.clock(), "\t", "add missing feature values"


        for postid in postids:
            #print postid, abspositions[(threadid,postid)]
            if not (threadid,postid) in cosinesimilaritiesthread:
                cosinesimilaritiesthread[(threadid,postid)] = 0.0
            if not (threadid,postid) in cosinesimilaritiestitle:
                cosinesimilaritiestitle[(threadid,postid)] = 0.0
            if not (threadid,postid) in responsecounts:
                # don't store the counts for the openingpost
                #print "postid not in responsecounts", postid, "opening post:", openingpost_for_thread[threadid]
                responsecounts[(threadid,postid)] = 0
            #else:
                #sel = 0
                #if postid in selected_by_focususer:
                #    sel = 1
            votes_for_this_post = 0
            if postid in postvotes_for_this_thread:
                votes_for_this_post = postvotes_for_this_thread[postid]

        #print time.clock(), "\t", "add feature values to columns for all threads"

        addvaluestocolumnsoverallthreads(threadids, "threadid")
        addvaluestocolumnsoverallthreads(postids_dict, "postid")

        addvaluestocolumnsoverallthreads(abspositions, "abspos")
        addvaluestocolumnsoverallthreads(relpositions, "relpos")
        addvaluestocolumnsoverallthreads(responsecounts, "noresponses")
        addvaluestocolumnsoverallthreads(upvotecounts, "noofupvotes")
        addvaluestocolumnsoverallthreads(cosinesimilaritiesthread, "cosinesimwthread")
        addvaluestocolumnsoverallthreads(cosinesimilaritiestitle, "cosinesimwtitle")
        addvaluestocolumnsoverallthreads(wordcounts, "wordcount")
        addvaluestocolumnsoverallthreads(uniquewordcounts, "uniquewordcount")
        addvaluestocolumnsoverallthreads(typetokenratios, "ttr")
        addvaluestocolumnsoverallthreads(relpunctcounts, "relpunctcount")
        addvaluestocolumnsoverallthreads(avgwordlengths, "avgwordlength")
        addvaluestocolumnsoverallthreads(avgsentlengths, "avgsentlength")
        addvaluestocolumnsoverallthreads(relauthorcountsinthreadforpost,"relauthorcountsinthread")



        #else:
        #    print "WARNING:", threadid, postid, "missing cosine sim", bodies[(threadid,postid)]
        #print threadid, postid, relpositions[(threadid,postid)], responsecounts[(threadid,postid)], wordcounts[(threadid,postid)], uniquewordcounts[(threadid,postid)], typetokenratios[(threadid,postid)]
        #print threadid, postid


columns_std = dict()

featnames = ("threadid","postid","abspos","relpos","noresponses","noofupvotes","cosinesimwthread","cosinesimwtitle","wordcount","uniquewordcount","ttr","relpunctcount","avgwordlength","avgsentlength","relauthorcountsinthread")

print time.clock(), "\t", "standardize feat values"


for featurename in featnames:
    columndict = columns[featurename]

    columndict_with_std_values = columndict
    if featurename != "postid" and featurename != "threadid":
        columndict_with_std_values = standardize_values(columndict,featurename)
    columns_std[featurename] = columndict_with_std_values


#featfile.write("threadid\tpostid\tabspos\trelpos\tnoresponses\tnoofupvotes\tcosinesimwthread\tcosinesimwtitle\twordcount\t"
#           "uniquewordcount\tttr\trelpunctcount\tavgwordlength\tavgsentlength"
           #"\treadability"
#           "\trelauthorcountsinthread"
           #"\tnrofvotes
#           "\n")

featfile = open(featfilename,'w')

print time.clock(), "\t", "print feature file"


for featname in featnames:
    featfile.write(featname+"\t")
featfile.write("\n")

# DON'T STANDARDIZE:

#for (threadid,postid) in threadids:
#    for featname in featnames:
#        columndict = columns[featname]
        #print featname, column_std
#        if (threadid,postid) in columndict:
#            featfile.write(str(columndict[(threadid,postid)])+"\t")
#        else:
#            featfile.write("NA\t")
#    featfile.write("\n")

# DO STANDARDIZE:


for (threadid,postid) in threadids:
    for featname in featnames:
        columndict_std = columns_std[featname]
        #print featname, columndict_std
        if (threadid,postid) in columndict_std:
            featfile.write(str(columndict_std[(threadid,postid)])+"\t")
        else:
            featfile.write("NA\t")
            print "not in columndict for feature", featname, ":", threadid, postid
    featfile.write("\n")


featfile.close()