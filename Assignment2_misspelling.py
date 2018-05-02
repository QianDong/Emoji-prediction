# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 00:00:21 2018

"""

import math
import numpy as np
import re
#import numpy as np
#import torch
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

lm = WordNetLemmatizer()

#data = np.loadtxt('train_us_semeval18_tweets.json.text', dtype='str',encoding="utf8")
text = open('train_us_semeval18_tweets.json.text', encoding="utf8", newline='\n')
#for row in data:
label = open('train_us_semeval18_tweets.json.labels', encoding="utf8")
#
test = open('us_trial.text', encoding="utf8",newline='\n')
#
test_label = open('us_trial.labels', encoding="utf8")


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

all_test_lines = []
for row in test:
    row = row.lower()
    all_test_lines.append(row)

test_labels = []
for row in test_label:
    line = row.split()
    number = int(line[0])
    test_labels.append(number)
print('finished reading')


text.close()
label.close()
test.close()

#===================
bigram_line = []
all_bigram = []
last_word = []
all_tokens=[]

re_sub_str = '[^@a-zA-Z\ \#]'
#re_sub_str = '[^a-zA-Z\ \!\?\-]'

# =========== all bigrams ==========
for line in text_in_lines:     
    this_line_token = re.sub(re_sub_str,' ', line )                        
    this_line_token = word_tokenize(this_line_token)
    this_line_bool = [True] * len(this_line_token) 
    for index,token in enumerate(this_line_token):
        if token == '@':
            this_line_bool[index] = False
            if index < len(this_line_token)-1:
                this_line_bool[index+1] = False
        if token == '#':
            this_line_bool[index] = False
            if index < len(this_line_token)-1:
                this_line_bool[index+1] = False    
        if len(str(token)) > 15:
            this_line_bool[index] = False 
    this_line_token = list( np.array(this_line_token)[this_line_bool] )         
    this_line_token = ['<s>'] + this_line_token + ['</s>']  
    # ============ for predict emoji ===========
    last_word.append(this_line_token[-2])
    # ======= generate bigram based on each line ======
    bi_for_this_line = [ this_line_token[i]+' '+this_line_token[i+1]  for i in range(len(this_line_token) -1) if this_line_token[i]!='️' and this_line_token[i+1]!='️']
    bigram_line.append(bi_for_this_line)
    for word in this_line_token:
        all_tokens.append(word)
all_tokens = [token for token in all_tokens if token != '️' ]
print('tokenize finished')

# ========== bigram ===========
for line in bigram_line:
    for bi in line:
        all_bigram.append(bi)
print('bigram finished')

#=============================
# 统计词频
token_count = Counter( token for token in all_tokens )
token_dict = dict(token_count)
bigram_count = Counter( str(bi) for bi in all_bigram ) 
bigram_dict = dict(bigram_count)
print(bigram_count.most_common(3))


# ============= probability ==================
def cal_P(bigram):
    s = bigram.split(' ')
    numerator = bigram_dict[bigram]
    denumerator = token_dict[s[0]]
    return numerator/denumerator

# =========== add-k smoothing =================
k = 0.0001
#V = len(all_tokens)-len(token_dict)
V= len(token_dict)
lambd = [0.6,0.4]

def cal_P_add_k(bigram):
    s = bigram.split(' ')
    numerator = bigram_dict[bigram] + k
    denumerator = token_dict[s[0]] + k*V
    return numerator/denumerator

# =========== interpolation ==================
def cal_P_interpolation(bigram):
    s = bigram.split(' ')
    numerator = bigram_dict[bigram]
    denumerator = token_dict[s[0]]
    unigram_p = token_dict[s[1]] / V 
    p_list = [numerator/denumerator, unigram_p]
    return np.dot(p_list,lambd)

# =============================================
p_dict_bigram = {bigram:cal_P(bigram) for bigram in bigram_dict.keys() }    
p_dict_bigram_add_k = {bigram:cal_P_add_k(bigram) for bigram in bigram_dict.keys() }
p_dict_bigram_interpolation =  {bigram:cal_P_interpolation(bigram) for bigram in bigram_dict.keys() }
print('p add-k finished')

# ======== test data ===============
test_last_word = []
all_test_token = []
test_bigram_line = []
test_lines = all_test_lines[:1000]
for line in test_lines:
    this_line_token = re.sub(re_sub_str,'', line )     
    this_line_token = word_tokenize(this_line_token)
    this_line_bool = [True] * len(this_line_token) 
    for index,token in enumerate(this_line_token):
        if token == '@':
            this_line_bool[index] = False
            if index < len(this_line_token)-1:
                this_line_bool[index+1] = False
        if token == '#':
            this_line_bool[index] = False    
            if index < len(this_line_token)-1:
                this_line_bool[index+1] = False   
        if len(str(token)) > 15:
            this_line_bool[index] = False 
    this_line_token = list( np.array(this_line_token)[this_line_bool] )         
    this_line_token = ['<s>'] + this_line_token + ['</s>']  
    # ============ for predict emoji ===========
    test_last_word.append(this_line_token[-2])
    # ======= generate bigram based on each line ======
    bi_for_this_line = [ this_line_token[i]+' '+this_line_token[i+1]  for i in range(len(this_line_token) -1) if this_line_token[i]!='️' and this_line_token[i+1]!='️']
    test_bigram_line.append(bi_for_this_line)

test_bigram = []    
for line in test_bigram_line:
    for bi in line:
        test_bigram.append(bi)
print('bigram finished')

# =============== levenshteinDistance ==============
# https://stackoverflow.com/questions/2460177/edit-distance-in-python
min_LD = 5   # only correcting words if the LevenshteinDistance is less than min_LD
def levenshteinDistance(w1, w2):
    if len(w1) > len(w2):
        w1, w2 = w2, w1
    if len(w2) - len(w1) > min_LD:
        return min_LD 
    distances = range(len(w1) + 1)
    for index2, char2 in enumerate(w2):
        L_distances = [index2+1]
        for index1, char1 in enumerate(w1):
            if char1 == char2:
                L_distances.append(distances[index1])
            else:
                L_distances.append(1 + min((distances[index1], distances[index1 + 1], L_distances[-1])))
        distances = L_distances
    if len(w1) == len(w2) and distances[-1] == 1:
        return 0.5   # make sure the 'swap-letter' word is the most possible word
    return distances[-1]


# =========================================
#       correcting spelling mistakes
# =========================================
N = len(test_bigram)  
# ========= calculate pp interpolation ====
pp_interpolation = 0
for bi in test_bigram:
    bis = bi.split(' ')
    if bi in bigram_dict.keys():
        p = p_dict_bigram_interpolation[bi]
    elif bis[1] in token_dict.keys():
        p = token_dict[bis[1]]/ len(bigram_dict) 
    else:
        p = 1/V
    pp_interpolation += -1/N * math.log(p)
print(pp_interpolation)   

'''
pp_interpolation = 0
for bi in test_bigram:
    bis = bi.split(' ')
    if bi in bigram_dict.keys():
        p = p_dict_bigram_interpolation[bi]
    elif bis[1] in token_dict.keys():
        p = token_dict[bis[1]]/ len(bigram_dict) 
    else:
        distance_dict = {levenshteinDistance(bis[1],word):word for word in token_dict.keys()}
        correct_word_s = min(distance_dict.keys())    
        correct_word = distance_dict[correct_word_s]
        print(bis[1],correct_word)
        p = token_dict[correct_word] / len(bigram_dict) 
        #p = 0.00001
    pp_interpolation += -1/N * math.log(p)
print(pp_interpolation)       
'''

# =========================================
#       introduce 3 types of mistakes
# =========================================
test_last_word = []
all_test_token = []
normalized_test_lines = []
for line in all_test_lines:
    this_line_token = re.sub(re_sub_str,' ', line )     
    this_line_token = word_tokenize(this_line_token)
    this_line_bool = [True] * len(this_line_token) 
    for index,token in enumerate(this_line_token):
        if token == '@':
            this_line_bool[index] = False
            if index < len(this_line_token)-1:
                this_line_bool[index+1] = False
        if token == '#':
            this_line_bool[index] = False    
            if index < len(this_line_token)-1:
                this_line_bool[index+1] = False   
        if len(str(token)) > 15:
            this_line_bool[index] = False     
    this_line_token = list( np.array(this_line_token)[this_line_bool] ) 
    normalized_test_lines.append(this_line_token)
normalized_test_lines = [line for line in normalized_test_lines if line != []]    

for i in range(len(normalized_test_lines)):
    if len(normalized_test_lines[i]) == 0:
        print (i,len(normalized_test_lines[i]))

# ========================================
def permute(w):
    permuted_word = list(w)
    permuted_word[-1] = w[-2]
    permuted_word[-2] = w[-1]
    word = ''.join(s for s in permuted_word)
    return word

def mistake(w):
    word = list(w)
    if word[-1]!='z':
        word[-1] = 'z'
    else:
        word[-1] = 'x'
    word = ''.join(s for s in word)
    return word

def missing(w):
    return w[:-1]

# ========================================
#     generate wrong words    
# ========================================
wrong_word = []
original = []
for line in normalized_test_lines[:1000]:
    for word in line:
        original.append(word)
        if len(word) < 5:
            wrong_word.append(word)
        elif len(word) < 8:
            wrong_word.append(mistake(word))
        elif len(word) < 12:
            wrong_word.append(permute(word))
        else: 
            wrong_word.append(missing(word))

# ========================================
#     correcting wrong words    
# ========================================
correcting = []
for word in wrong_word:
    if word not in  token_dict.keys():
        l = len(word)
        similar_words = [wo for wo in token_dict.keys() if abs(len(wo)-l)< 2 ]
        distance_dict = {levenshteinDistance(word,w):w for w in similar_words }
        correct_word_s = min(distance_dict.keys())    
        correct_word = distance_dict[correct_word_s]
        print(word,correct_word)
        correcting.append(correct_word)
    else:
        correcting.append(word)
    
correct_count = 0
for i in range(len(wrong_word)-1) :
    if original[i] == correcting[i]:
        correct_count += 1
print('accuracy:',correct_count*1.0/len(original))


