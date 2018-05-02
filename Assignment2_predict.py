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

#data = np.loadtxt('train_us_semeval18_tweets.json.text', dtype='str',encoding="utf8")
text = open('train_us_semeval18_tweets.json.text', encoding="utf8", newline='\n')
#for row in data:
label = open('train_us_semeval18_tweets.json.labels', encoding="utf8")
#
test = open('us_trial.text', encoding="utf8")
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

test_lines = []
for row in test:
    row = row.lower()
    test_lines.append(row)

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


re_sub_str = '[^a-zA-Z\ \!\?]'
#re_sub_str = '[.]'

def cal_trigrams(text_in_lines):
    trigram_line = []
    last_words = []
    all_tokens = []
    for line in text_in_lines:     
        this_line_token = re.sub(re_sub_str,'', line )                         
        this_line_token = word_tokenize(this_line_token)    
        this_line_token = ['<s>'] + this_line_token + ['</s>']  
        # ============ for predict emoji ===========
        last_words.append(this_line_token[-3:-1])
        # ======= generate bigram based on each line ======
        tri_for_this_line = [this_line_token[i]+' '+this_line_token[i+1]+' '+this_line_token[i+2] for i in range(len(this_line_token) -2) if this_line_token[i]!='️' and this_line_token[i+1]!='️' and this_line_token[i+2]!='️']
        trigram_line.append(tri_for_this_line)
        for word in this_line_token:
            all_tokens.append(word)
    all_tokens = [token for token in all_tokens if token != '️' ]
    return trigram_line,all_tokens,last_words

# =========== all bigrams ==========

trigram_line, all_tokens, last_words = cal_trigrams(text_in_lines)
print('tokenize finished')

# ========== bigram ===========
all_trigram = []
for line in trigram_line:
    for tri in line:
        all_trigram.append(tri)
print('trigram finished')

#=============================
# 统计词频
token_count = Counter( token for token in all_tokens )
token_dict = dict(token_count)
trigram_count = Counter( str(tri) for tri in all_trigram ) 
trigram_dict = dict(trigram_count)
print(trigram_count.most_common(3))


# ============= probability ==================
def cal_P(trigram):
    s = trigram.split(' ')
    numerator = trigram_dict[trigram]
    denumerator = token_dict[s[0]]
    return numerator/denumerator

# =========== add-k smoothing =================
k = 0.0001
#V = len(all_tokens)-len(token_dict)
V= len(token_dict)
lambd = [0.6,0.4]

def cal_P_add_k(trigram):
    s = trigram.split(' ')
    numerator = trigram_dict[trigram] + k
    denumerator = token_dict[s[0]] + k*V
    return numerator/denumerator

# =========== interpolation ==================
def cal_P_interpolation(trigram):
    s = trigram.split(' ')
    numerator = trigram_dict[trigram]
    denumerator = token_dict[s[0]]
    unigram_p = token_dict[s[1]] / V 
    p_list = [numerator/denumerator, unigram_p]
    return np.dot(p_list,lambd)

# =============================================
p_dict_trigram = {trigram:cal_P(trigram) for trigram in trigram_dict.keys() }    
p_dict_trigram_add_k = {trigram:cal_P_add_k(trigram) for trigram in trigram_dict.keys() }
p_dict_trigram_interpolation =  {trigram:cal_P_interpolation(trigram) for trigram in trigram_dict.keys() }
print('p add-k finished')

# ======== test data ===============

test_trigram_line,all_test_token,test_last_words=cal_trigrams(test_lines)

test_trigram = []    
for line in test_trigram_line:
    for tri in line:
        test_trigram.append(tri)
print('test trigram finished')

# ========= calculate pp probability ==============
pp = 0
N = len(test_trigram)
for tri in test_trigram:
    if tri in trigram_dict.keys():
        p = p_dict_trigram[tri]
    else:
        p = 0.00001
    pp += -1/N * math.log(p) 
print(pp)

    
# ========= calculate pp add-k ==============
pp_add_k = 0
for tri in test_trigram:
    if tri in trigram_dict.keys():
        p = p_dict_trigram_add_k[tri]
    else:
        p = 0.00001
    pp_add_k += -1/N * math.log(p)
print(pp_add_k)

 # ========= calculate pp interpolation ==============
pp_interpolation = 0
for tri in test_trigram:
    tris = tri.split(' ')
    if tri in trigram_dict.keys():
        p = p_dict_trigram_interpolation[tri]
    elif str(tris[1])+' '+str(tris[2]) in bigram_dict.keys():  # from bigram, calculate that first
        p = p_dict_bigram_interpolation[str(tris[1])+' '+str(tris[2])]    
    elif str(tris[2]) in token_dict.keys():
        p = token_dict[tris[2]]/ len(trigram_dict) 
    else:
        p = 0.00001
    pp_interpolation += -1/N * math.log(p)
print(pp_interpolation)       


# ========= train set emoji bigram ==================
emoji_trigram = []
for i in range(len(last_words)):
    emoji_trigram.append([last_words[i],labels[i]])

   
# =========== add-k smoothing =================
train_lastword_emoji_dict = { str(tri[0][0])+' '+str(tri[0][1]):[0]*20 for tri in emoji_trigram }   
for tri in emoji_trigram:
    bi = str(tri[0][0])+' '+str(tri[0][1])
    emoji = tri[1]
    temp_list = []
    temp_list[:] = train_lastword_emoji_dict[bi]   # bi[1] is type_no of emoji
    temp_list[emoji]  += 1
    train_lastword_emoji_dict.update( { bi : temp_list } )
    
train_lastword_emoji_predict = { tri:np.argsort(p_list)[::-1] for tri,p_list in train_lastword_emoji_dict.items()}
 

# ================= predict ==================
# Counter(emoji for emoji in labels).most_common(3)
test_prediction =  []
for words in test_last_words:
    words = str(words[0])+' '+str(words[1])
    if words in train_lastword_emoji_predict.keys():
        p = train_lastword_emoji_predict[words][0]
    else:
        p = 12
    test_prediction.append(p)
  
# =============   F1  ====================
F1 = {}
for emoji in [0,1,2,3,4,5,7,10,11,12,13,15,18]:  # missing 6,8,9,14,16,17,19:
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(test_labels)):
        if test_labels[i] == emoji and test_prediction[i] == emoji:
            tp += 1
        elif test_labels[i] == emoji and test_prediction[i] != emoji:
            fn += 1
        elif test_prediction[i] == emoji and test_labels[i] != emoji:
            fp += 1
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2 * ( precision * recall / (precision + recall) )
    F1.update({emoji:f1})
    




