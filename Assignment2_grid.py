# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 19:15:19 2018

@author: Qian
"""
import math
import numpy as np
import re
import random
#import numpy as np
#import torch
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import KFold

lm = WordNetLemmatizer()

#data = np.loadtxt('train_us_semeval18_tweets.json.text', dtype='str',encoding="utf8")
text = open('train_us_semeval18_tweets.json.text', encoding="utf8", newline='\n')
#for row in data:
label = open('train_us_semeval18_tweets.json.labels', encoding="utf8")


all_words = []
text_in_lines = []

for row in text: 
    row = row.lower()
    text_in_lines.append(row)    

labels = []
for row in label:
    line = row.split()
    number = int(line[0])
    labels.append(number)

text.close()
label.close()

# ============================================

def all_train_test_index(fold, X):
    kf = KFold(n_splits = fold, shuffle = True)
    all_train_index = []
    all_test_index = []
    for train_index, test_index in kf.split(X):   # each index is a row of [userId, movieId] in X
        all_train_index.append(train_index)
        all_test_index.append(test_index)
    return all_train_index, all_test_index



# ====================================
#   create subset's index
# ====================================
index = list(range(len(text_in_lines)))
random.shuffle(index)
index = index[:10000]
train_index, test_index = all_train_test_index(5,np.array(text_in_lines)[index])



#===================
#re_sub_str = '[^@a-zA-Z\ \#]'
re_sub_str = '[^a-zA-Z\ \!\?\-]'   #f.x. sista-to-be



max_token_len = 15

def cal_bigrams(text_in_lines):
    bigram_line = []
    last_word = []
    all_tokens = []
    for line in text_in_lines:     
        this_line_token = re.sub(re_sub_str,' ', line )                         
        this_line_token = word_tokenize(this_line_token)  
        
        this_line_token = [token for token in this_line_token if len(token) < max_token_len] 
        
        this_line_token = ['<s>'] + this_line_token + ['</s>']  
        # ============ for predict emoji ===========
        last_word.append(this_line_token[-2])
        # ======= generate bigram based on each line ======
        bi_for_this_line = [ this_line_token[i]+' '+this_line_token[i+1]  for i in range(len(this_line_token) -1) if this_line_token[i]!='️' and this_line_token[i+1]!='️']
        bigram_line.append(bi_for_this_line)
        for word in this_line_token:
            all_tokens.append(word)
    all_tokens = [token for token in all_tokens if token != '️' ]
    return bigram_line,all_tokens,last_word

# =========== add-k smoothing =================
def cal_P_add_k(bigram,k):
    s = bigram.split(' ')
    numerator = bigram_dict[bigram] + k
    denumerator = token_dict[s[0]] + k*V
    return numerator/denumerator

# =========== interpolation ==================
def cal_P_interpolation(bigram,lambd):
    s = bigram.split(' ')
    numerator = bigram_dict[bigram]
    denumerator = token_dict[s[0]]
    unigram_p = token_dict[s[1]] / V 
    p_list = [numerator/denumerator, unigram_p]
    return np.dot(p_list,lambd)

'''
# ====================================
#   Lambda p-interpolation
# ====================================
lambd_1 = np.arange(0.0,1.0,0.1)
pp_add_k_dict = {}
pp_interpolation_dict = {}
for i in lambd_1:
    pp_add_k_list = []
    pp_interpolation_list = []
    for trial in range(5):
        X_train = np.array(text_in_lines)[[train_index[trial]]]
        y_train = np.array(labels)[[train_index[trial]]]
        X_test = np.array(text_in_lines)[[test_index[trial]]]
        y_test = np.array(labels)[[test_index[trial]]]
        
        this_bigram_line, this_trial_tokens, this_trial_last_word = cal_bigrams(X_train)
        this_test_bigram_line, this_test_tokens, this_test_last_word = cal_bigrams(X_test)
        # ========== bigram ===========
        bigram_of_this_trial = []
        for line in this_bigram_line:
            for bi in line:
                bigram_of_this_trial.append(bi)
        
        # =============================
        # 统计词频
        token_count = Counter( token for token in this_trial_tokens )
        token_dict = dict(token_count)
        bigram_count = Counter( str(bi) for bi in bigram_of_this_trial ) 
        bigram_dict = dict(bigram_count)
        #print(bigram_count.most_common(3))
                
        # ====================        
        V= len(token_dict)
        #p_dict_bigram_add_k = {bigram:cal_P_add_k(bigram) for bigram in bigram_dict.keys() }    
        p_dict_bigram_interpolation =  {bigram:cal_P_interpolation(bigram,[i,1-i]) for bigram in bigram_dict.keys() }

        # ====================
        test_bigram = []    
        for line in this_test_bigram_line:
            for bi in line:
                test_bigram.append(bi)
        
        # ========= calculate pp interpolation ==============
        N = len(test_bigram)
        pp_interpolation = 0
        for bi in test_bigram:
            bis = bi.split(' ')
            if bi in bigram_dict.keys():
                p = p_dict_bigram_interpolation[bi]
            elif bis[1] in token_dict.keys():
                p = token_dict[bis[1]]/ len(bigram_dict) 
            else:
                p = 0.00001
            pp_interpolation += -1/N * math.log(p)
        pp_interpolation_list.append(pp_interpolation)
    
    #pp_add_k_dict.update( {i: sum(pp_add_k_list)/len(pp_add_k_list) } )
    pp_interpolation_dict.update( {i: sum(pp_interpolation_list)/len(pp_interpolation_list) }  )
    
    #print(i,sum(pp_add_k_list)/len(pp_add_k_list) )
    print(i,sum(pp_interpolation_list)/len(pp_interpolation_list) )
''' 

# ====================================
#   add-k  test k
# ====================================
k_list = [0.0001,0.001,0.01,0.1]
pp_add_k_dict = {}
pp_interpolation_dict = {}
for k in k_list:
    pp_add_k_list = []
    for trial in range(5):
        X_train = np.array(text_in_lines)[[train_index[trial]]]
        y_train = np.array(labels)[[train_index[trial]]]
        X_test = np.array(text_in_lines)[[test_index[trial]]]
        y_test = np.array(labels)[[test_index[trial]]]
        
        this_bigram_line, this_trial_tokens, this_trial_last_word = cal_bigrams(X_train)
        this_test_bigram_line, this_test_tokens, this_test_last_word = cal_bigrams(X_test)
        # ========== bigram ===========
        bigram_of_this_trial = []
        for line in this_bigram_line:
            for bi in line:
                bigram_of_this_trial.append(bi)
        
        # =============================
        # 统计词频
        token_count = Counter( token for token in this_trial_tokens )
        token_dict = dict(token_count)
        bigram_count = Counter( str(bi) for bi in bigram_of_this_trial ) 
        bigram_dict = dict(bigram_count)
        #print(bigram_count.most_common(3))
                
        # ====================        
        V= len(token_dict)
        p_dict_bigram_add_k = {bigram:cal_P_add_k(bigram,k) for bigram in bigram_dict.keys() }    

        # ====================
        test_bigram = []    
        for line in this_test_bigram_line:
            for bi in line:
                test_bigram.append(bi)

        # ========= calculate pp add-k ==============
        N = len(test_bigram)
        pp_add_k = 0
        for bi in test_bigram:
            if bi in bigram_dict.keys():
                p = p_dict_bigram_add_k[bi]
            else:
                p = 0.00001
            pp_add_k += -1/N * math.log(p)
        pp_add_k_list.append(pp_add_k)
   
    pp_add_k_dict.update( {k: sum(pp_add_k_list)/len(pp_add_k_list) } )
    
    print(k,sum(pp_add_k_list)/len(pp_add_k_list) )


















