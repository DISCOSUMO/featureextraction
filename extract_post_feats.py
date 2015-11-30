# python extract_feats.py ../dataconversion/reddit_data/per_subreddit/NoStupidQuestions/2nw7wr.xml

# Feature extraction.
# + (a) popularity (#upvotes, #responses)
# + (b) representativeness for the thread (cosine similarity for tf-idf weighted vector representations of post and thread/title)
# (c) readability (length,type-token ratio, average sentence length, punctuation marks, Flesch(-Douma) statistic for Readability Ease))
# + (d) prominence of author (# posts total, # posts in category, # posts in thread)
# (e) position in the thread (relative, absolute)


import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import re
import math
import fileinput
from collections import defaultdict
import xml.etree.ElementTree as ET
#from xml.dom import minidom

rootdir = sys.argv[1]
rootdir = re.sub("[^\/]+\/[a-zA-Z0-9]+\.xml","",rootdir)
sys.stderr.write("Rootdir: "+rootdir)


upvotecounts = defaultdict(int); # key is postid, value is # of upvotes
responsecounts = defaultdict(int); # key is postid, value is # of replies
termvectors = defaultdict(dict) # key is postid, value is dict with term -> termcount for post
termvectorforthread = dict() # key is term, value is termcount for full thread
termvectorfortitle = dict() # key is term, value is termcount for title
cosinesimilaritiesthread = dict() # key is postid, value is cossim with term vector for complete thread
cosinesimilaritiestitle = dict() # key is postid, value is cossim with term vector for title
uniquewordcounts = defaultdict(int) #key is postid, value is unique word count in post
wordcounts = defaultdict(int) #key is postid, value is word count in post
typetokenratios = defaultdict(float) # key is postid, value is type-token ratio in post
authorcountsinthread = dict() # key is authorid, value is number of posts by author in this thread
authorcountsincategory = defaultdict(int) # key is authorid, value is number of posts by author in this category

postsperauthor = defaultdict(list) # key is authorid, value is list of postids
bodies = defaultdict(str) #key is postid, value is content of post



def tokenize(t):
    text = t.lower()
    text = re.sub("[^a-zA-Z0-9- ']", "", text)
#    print text
    words = text.split()
    return words



def printTopNPosts(d,n):
    i=0
    for postid in sorted(d, key=d.get, reverse=True):
        i = i+1
        if (i > n):
            break
        else:
            print postid, d[postid]
            #print postid, bodies[postid], d[postid]


def getTopNAuthors(d,n,minnrofposts):
    topauthors = list()
    i=0
    print "    Top"+str(n)+" authors:"
    for author in sorted(d, key=d.get, reverse=True):
        i = i+1
        print "\t" +author, d[author]
        if (i > n or d[author]<minnrofposts):
            break
        else:
            topauthors.extend([author])
    return topauthors

def getAuthorCountsforCategory(rootdir,category):
    filename = rootdir+"/"+category+"/authors.list"
#    print filename
    authorcounts = dict()
    if (not os.path.isfile(filename)):
        sys.stderr.write("Error: "+filename+" does not exist!")
    else:
        for line in fileinput.input([filename]):
            match = re.search("<author>(.*)</author>", line)
            authorname = match.group(1)
            #print authorname
            if authorname in authorcounts.keys():
                authorcounts[authorname] = authorcounts[authorname] +1
            else:
                authorcounts[authorname] = 1
    return authorcounts


def normalizeVector(termvector):
    sumofsquares=0
    for component in termvector.values():
        sumofsquares = sumofsquares + component*component
    vectorlength = math.sqrt(sumofsquares)
    for term in termvector.keys():
        normvalue = termvector[term]/vectorlength
        termvector[term] = normvalue
    return termvector

def cosineSimilarity(v1,v2):
    innerproduct = 0
    if (len(v1) != len(v2)):
        sys.stderr.write("Warning: length of two vectors not equal!")
    else:
        for c1 in v1.keys():
            if (c1 not in v2.keys()):
                sys.stderr.write("Warning: vectors do not contain the same terms! "+c1)
            else:
                innerproduct = innerproduct + v1[c1]*v2[c1]
    return innerproduct

def logtransformandnormalize(termvector):
    for vterm in termvector:
        tcount = termvector[vterm]
        #print(vterm,tcount)
        if tcount == 0:
            tweight = 0
        else:
            tweight = 1+math.log(tcount)
        termvector[vterm] = tweight

    termvector= normalizeVector(termvector)
    return termvector


tree = ET.parse(sys.argv[1])
root = tree.getroot()

for thread in root:
    category = thread.find('category').text
    title = thread.find('title').text
    titlewords = tokenize(title)
    for tw in titlewords:
        if tw in termvectorfortitle:
            termvectorfortitle[tw] += 1
        else:
            termvectorfortitle[tw] =1

    authorcountsincategory = getAuthorCountsforCategory(rootdir,category)

for posts in thread.findall('posts'):
    for post in posts.findall('post'):
        postid = post.get('id')

        parentid = post.find('parentid').text
        if parentid in responsecounts.keys():
            responsecounts[parentid] = responsecounts[parentid] +1
        else:
            responsecounts[parentid] = 1

        upvotes = post.find('upvotes').text
        upvotecounts[postid] = int(upvotes)
        
        author = post.find('author').text
        if author in postsperauthor.keys():
            postsperauthor[author].extend([postid])
        else:
            postsperauthor[author] = [postid]


        body = post.find('body').text
        if body is None:
            continue
        else:
            bodies[postid] = body

            words = tokenize(body)
            wc = len(words)
            wordcounts[postid] = wc
            for word in words:
                if word in termvectorforthread:# dictionary over all posts
                    termvectorforthread[word] += 1
                else:
                    termvectorforthread[word] = 1
                    
                if word in termvectors[postid].keys(): # dictionary per post
                    termvectors[postid][word] += 1
                else:
                    termvectors[postid][word] = 1


# add zeroes for titleterms that are not in the thread vector
for titleword in termvectorfortitle:
    if titleword not in termvectorforthread:
        termvectorforthread[titleword]=0

# add zeroes for terms that are not in the title vector:
for word in termvectorforthread:
    if word not in termvectorfortitle:
        termvectorfortitle[word]=0

# add zeroes for terms that are not in the post vector:
for postid in termvectors.keys():
    termvectorforpost = termvectors[postid]
    for word in termvectorforthread:
        if word not in termvectorforpost:
            termvectorforpost[word] = 0
    termvectors[postid] = termvectorforpost


print("\nPosts that are the most popular:")
print("- Posts with most upvotes:")
printTopNPosts(upvotecounts,5)

print("- Posts with most responses:")
printTopNPosts(responsecounts,5)


termvectorforthread = logtransformandnormalize(termvectorforthread)
termvectorfortitle = logtransformandnormalize(termvectorfortitle)

#for term in termvectorforthread:
#    print (term,termvectorforthread[term])

for postid in termvectors.keys():
    uniquewordcount = 0
    termvectorforpost = termvectors[postid]
    termvectorforpost = logtransformandnormalize(termvectorforpost)
    termvectors[postid] = termvectorforpost

    for term in termvectorforpost:
        uniquewordcount = uniquewordcount+1

    uniquewordcounts[postid] = uniquewordcount
    typetokenratio = float(uniquewordcount)/float(wordcounts[postid])
    typetokenratios[postid] = typetokenratio

#    for term in termvectorforthread:
#        print(postid,term,termvectors[postid][term])
    cossimthread = cosineSimilarity(termvectors[postid],termvectorforthread)
    cossimtitle = cosineSimilarity(termvectors[postid], termvectorfortitle)
    cosinesimilaritiesthread[postid] = cossimthread
    cosinesimilaritiestitle[postid] = cossimtitle
#    print(postid,cossim)

print("\nPosts that are the most representative for the thread:")
print("- centroid similarity:")
printTopNPosts(cosinesimilaritiesthread,5)
print("- title similarity:")
printTopNPosts(cosinesimilaritiestitle,5)


print("\nPosts that have the highest text quality:")
print("- the highest number of unique words:")
printTopNPosts(uniquewordcounts,5)

#print("- the highest type-token ratio:")
#printTopNPosts(typetokenratios,5)


print("\nPosts that are by the most prominent users:")
print("- by users with the highest number of posts on the complete forum:")
print("- by users with the highest number of posts in the category:")

topauthorsincategory = getTopNAuthors(authorcountsincategory,5,2)
for author in topauthorsincategory:
    postsfromtopauthors = postsperauthor[author]
    for postid in postsfromtopauthors:
        print postid,author

print("- by users with the highest number of posts in the thread:")

for author in postsperauthor.keys():
    postcount = len(postsperauthor[author])
    authorcountsinthread[author] = postcount
    #print(author,str(postcount))

topauthorsinthread = getTopNAuthors(authorcountsinthread,5,2)
for author in topauthorsinthread:
    postsfromtopauthors = postsperauthor[author]
    for postid in postsfromtopauthors:
        print postid,author
