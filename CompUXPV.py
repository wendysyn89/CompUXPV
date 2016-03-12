#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


import logging
import sys
import os
from word2vec import Word2Vec, Sent2Vec, LineSentence
import gensim
import re
import csv
import utils, matutils
from collections import Counter
from collections import defaultdict
import codecs
import pymongo

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))

# try:
#     conn=pymongo.MongoClient()
#     print "Connected successfully!!!"
# except pymongo.errors.ConnectionFailure, e:
#    print "Could not connect to MongoDB: %s" % e
#
# # Define my mongoDB database
# db = conn.dataset
# # Define my collection where I'll insert my search
# amazon = db.amazon_review
# para = db.paragraph

# input_file = 'all.txt.gz'
input_file = 'all.txt.gz'
item_file= 'all_item_label.txt'
review_file= 'review.txt'

program = os.path.basename(sys.argv[0])
logging.info("finished running %s" % program)

'''learn word vector'''
def learn_word_vector():
    model = Word2Vec(LineSentence(input_file), size=100, window=5, sg=0, min_count=5, workers=8)
    #model.save(input_file + '.model')
    model.save_word2vec_format('all_item_label.model.bin', binary=True)
    model.save_word2vec_format(input_file + '.vec')

'''learn vector for review'''
def learn_p_vector():
    model = Sent2Vec(LineSentence(review_file), model_file=input_file + '.model')
    model.save_sent2vec_format(review_file + '.vec')

'''learn vector for item'''
def learn_item_vector():
    model = Sent2Vec(LineSentence(item_file), model_file=input_file + '.model')
    model.save_sent2vec_format(item_file + '.vec')


def read_sent(txt,result,sid):
    text_file = open("ce.txt", "r")
    lines = text_file.readlines()
    for item in lines:
        m = re.search(r"%*(\d+)%",item)#extract sent id
        sent_id= m.group(1)
        print sent_id

        """for checking fmeasure"""
        if ("sent_%i" %int(sent_id))==result[0][0]:
            print result[0][0]
            if result[0][1]>0:
                with open("ce_result.txt", 'a') as f:
                    f.write(txt+item+str(result[0][1])+'\n'+'\n')
            # with open("docvec_sent_3.txt", 'a') as f:
            #     f.write(item+'\n'+row[3]+'\n'+str(result[0][1])+'\n'+'\n')

            # with open("cat4lsa_dbow.txt", 'a') as f:
            #     f.write(catid+'\n')


    return None


def get_best_ux2(result1,result2):
        scorelist=[]
        for item in result1:
            print item[0]
            print item[1]
            scorelist.append((item[0],item[2]))

        for item in result2:
            print item[0]
            print item[1]
            scorelist.append((item[0],item[2]))
        maximum= max(scorelist, key=lambda x: x[1])
        print scorelist
        print maximum[0]
        return maximum[0]

def get_best_ux(result):
    scorelist=[]
    idlist=[]
    uxdict={}
    #print result

    for items in result:
        scorelist.append(items[2])#get the similarity score
        idlist.append(items[0])#get the category id

    variance=scorelist[0]-scorelist[1] #check the variance for the first two item

    #print 'variance',variance

    """if the variance>0.2,direct get the first item as the best category"""
    if variance>0.2:
        #print 'large variance'
        cid=idlist[0]
        with open('doc2vecresult.txt', 'a') as f:
            f.write('large variance'+'\n')


    else:
        """if variance less than threshold,take the one with highest frequency"""
        #print 'calculate frequency'
        c=Counter(elem for elem in idlist)
        with open('subspace_lsi_em6.txt', 'a') as f:
            f.write(str(c)+'\n')
        #print c
        m = max(v for _, v in c.iteritems())
        r = [k for k, v in c.iteritems() if v == m]

        """if have similar frequency,take the one with higher similarity score"""

        if len(r)>1:
            with open('doc2vecresult.txt', 'a') as f:
                f.write('share same frequency,get the construct with highest max score'+'\n')
            #print 'share same frequency,get the construct with highest max score'
            uxdict= defaultdict( list )

            """group the category id in dictionary"""

            for v,w,x in result:
                if v in r:
                    uxdict[v].append(x)
            #print uxdict
            """take the one with maximum score"""

            for key,value in uxdict.items():
                uxdict[key]=max(value)
            cid= max(uxdict, key=uxdict.get)

        else:

            with open('ann4_doc2vecresult_details.txt', 'a') as f:
                f.write('get the construct with highest frequency'+'\n')
            #print 'get the construct with highest frequency'
            for item in r:
                cid=item
    #print 'cid',cid
    return cid

"""check similarity"""
def check_similar(new_last_line):
    text_file2 = open("all_item_label.txt.vec", "r")
    lines2 = text_file2.readlines()

    # new_last_line = ("bcd")

    #print new_last_line
    lines2[-1] = new_last_line
    open("all_item_label.txt.vec", 'w').writelines(lines2)

    model = gensim.models.Doc2Vec.load_word2vec_format('all_item_label.txt.vec', binary=False)
    return model.most_similar("sent_1028")

def get_details(item):
    # print item
    result=[]
    with open('item.csv', 'rb') as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            # print row[0]
            # print item[0]
            if row[0]==item[0]:
                #print 'here'
                sent_id= row[0]
                cid=row[1]
                txt=row[2]
                result.append((cid,txt,item[1]))
                #print result
                return cid,txt,item[1]


def get_result_list(result):
    resultlist=[]
    for i, item in enumerate(result):
        if i<6:
            #print i
            #print item
            a,b,c=get_details(item)
            resultlist.append((a,b,c))
            with open("ann4_doc2vecresult_details.txt", 'a') as f:
                f.write("item:::"+str(a)+'\t'+str(b)+'\t'+str(c)+'\n')
    return resultlist

result=[]
def get_ann_details(item):
    #print item
    sid= item.split()[0]
    with open('ann4.csv', 'rb') as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            # print row[0]
            # print item[0]
            if row[0]==sid:
                #print 'here'
                id= row[1]
                txt= row[2]
                with open("ann4_doc2vecresult_details.txt", 'a') as f:
                    f.write('\n'+"review:::"+str(txt)+'\t'+str(id)+'\n')
                return id,txt


def get_cid(idd):
    #print idd
    if idd=="1":
        aid=277
        return aid
    elif idd=="2":
        aid=278
        return aid
    elif idd=="3":
        aid=291
        return aid
    elif idd=="4":
        aid=343
        return aid
    elif idd=="5":
        aid=296
        return aid

def get_construct(idd):
        #print idd
        if idd=="277":
            an="Perceived Ease of Use"
            return an
        elif idd=="278":
            an="Perceived Usefulness"
            return an
        elif idd=="291":
            an="Affects towards Technology"
            return an
        elif idd=="296":
            an="Social Influence"
            return an
        elif idd=="343":
            an="Trust"
            return an

# text_file = open("ann4.txt.vec", "r")
#
# lines = text_file.readlines()
# for i,item in enumerate(lines):
#
#     if i>0:
#         actual_id,review=get_ann_details(item)
#         print 'review', review
#         acid= get_cid(actual_id)
#         sp= item.split()
#         sp[0]='sent_1028'
#         new_last_line=' '.join(sp)
#         result=check_similar(new_last_line)
#         r_list=get_result_list(result)
#         print 'r list',r_list
#         predict_id= get_best_ux(r_list)
#         print 'predict id',predict_id
#         print '\n'
#         with open("ann4_doc2vecresult_details_2.txt", 'a') as f:
#             f.write('\n'+"predicted id:::"+str(predict_id)+'\t'+'\n')
#         with open("ann4_doc2vec_result.txt", 'a') as f:
#             f.write('@'+str(predict_id)+'@'+'#'+str(acid)+'#'+str(review)+'\n')

#review_text= "This phone is good. I feel happy."


'''predict construct category'''
def get_category():
    result_list=[]

    #learn_item_vector()
    learn_p_vector()

    text_file = open("review.txt.vec", "r")
    lines = text_file.readlines()

    for i,item in enumerate(lines):
        if i>0:
            sp= item.split()
            sp[0]='sent_1028'
            new_review_text=' '.join(sp)
            print new_review_text
            result=check_similar(new_review_text)
            r_list=get_result_list(result)
            print 'r list',r_list
            predict_id= get_best_ux(r_list)
            print 'predict id',predict_id
            ux= get_construct(predict_id)
            print ux
            result_list.append(predict_id)
            result_list.append(ux)

    return result_list


'''##### Main function to predict construct category #####'''

"""Learn word vector"""
#learn_word_vector()

'''learn paragraph vector from file (e.g. measurement items)'''

#learn_p_vector()

"""Learn paragraph vector for review/query and get category"""

get_category()



