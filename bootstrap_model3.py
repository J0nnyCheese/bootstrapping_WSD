#!/usr/bin/env python
# coding: utf-8

from nltk.corpus import semcor
from nltk.tree import Tree
from nltk.corpus import stopwords
import pickle
import string
import numpy as np
from pandas import DataFrame
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import loader as ld


class custom_WSDInstance:
    def __init__(self, lemma, context,gold_key):
        self.lemma = lemma      # lemma of the word whose sense is to be resolved
        self.context = context  # lemma of all the words in the sentential context
        self.gold_key = gold_key
    def __str__(self):
        context = ' '.join(self.context)
        return '%s\n%s\n%s\n' % ( self.lemma, context, self.gold_key)

def remove_stopwords(context):
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []
    for w in context:
        if w not in stop_words:
            filtered_sentence.append(w)
    #print(filtered_sentence)
    return filtered_sentence

def create_df(data):
    data_arr = []
    for inst in data:
        str_ = ' '.join(word for word in inst.context)
        data_arr.append({'context': str_, 'gold_key': inst.gold_key})
    return DataFrame(data_arr)


def to_str_arr(word_2d_arr):
    result = []
    for arr in word_2d_arr:
        str_ = ' '.join(word for word in arr)
        result.append(str_)
    print(np.shape(result))
    return result


def train_clf(df,seed): #both df and seed are panda dataframe
    all_acc = []
    train_X = df['context'].values
    train_Y = df['gold_key'].values
    
    lr = LogisticRegression(multi_class='multinomial', max_iter = 800)
    vectorizer = CountVectorizer()
    pipe = Pipeline([
        ('count_vectorizer', vectorizer),
        ('classifier', lr)
    ])
    
    # Do resampling 12 times
    for i in range(1, 12):
        seed_X = seed['context'].values
        seed_Y = seed['gold_key'].values
        pipe.fit(seed_X, seed_Y)
        pred_Y = pipe.predict(train_X)

        idx = 0     
        # add instances to seed if correctly predicted
        for pred_y, gold_y in zip(pred_Y, train_Y):
            if pred_y == gold_y:
                seed = seed.append(df.loc[idx], ignore_index = True)
            idx = idx + 1
        curr_acc = accuracy_score(train_Y, pred_Y)
        all_acc.append(curr_acc)
    print(curr_acc)
    return pipe

def test_clf(pipe, df): #both df and seed are panda dataframe
    X = df['context'].values
    gold_Y = df['gold_key'].values
    
    pred_Y = pipe.predict(df['context'].values)
    acc = accuracy_score(gold_Y, pred_Y)
    
    print(acc)


def load_Semeval_xml():
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = ld.load_instances(data_f)
    dev_key, test_key = ld.load_key(key_f)
    
    dev_instances = {k:v for (k,v) in dev_instances.items() if k in dev_key}
    test_instances = {k:v for (k,v) in test_instances.items() if k in test_key}
    return dev_instances, dev_key, test_instances, test_key

def create_semeval_df(instances, keys):
    instances_arr = []
    keys_arr = []
    data_arr = []
    
    for k,v in instances.items():
        instances_arr.append(v)
    for k,v in keys.items():
        keys_arr.append(v)
        
    for (a_inst, a_key) in zip(instances_arr, keys_arr):
        str_ = ' '.join(word.decode("utf-8") for word in a_inst.context)
        data_arr.append({'context': str_, 'gold_key': a_key[0]})
    return DataFrame(data_arr)

def load_dev_test():
    dev_instances, dev_key, test_instances, test_key = load_Semeval_xml()
    dev_df = create_semeval_df(dev_instances, dev_key)
    test_df = create_semeval_df(test_instances, test_key)
    return dev_df, test_df

def retriv_from_disk(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def main():
    country = retriv_from_disk('./training_data/country_74.pkl')
    country_df = create_df(country)
    
    action = retriv_from_disk('./training_data/action_97.pkl')
    action_df = create_df(action)
    
    week = retriv_from_disk('./training_data/week_73.pkl')
    week_df = create_df(week)
    week_seed = week_df[:5].append(week_df.loc[68], ignore_index = True)
    
    world = retriv_from_disk('./training_data/world_150.pkl')
    world_df = create_df(world)
    
    deal = retriv_from_disk('./training_data/deal_37.pkl')
    deal_df = create_df(deal)
    
    print('Evaluate on training set')
    
    print('Word: Country')
    country_clf = train_clf(country_df, country_df[:5])
    print('Word: Action')
    action_clf = train_clf(action_df, action_df[:5])
    print('Word: Week')
    week_clf = train_clf(week_df, week_seed)
    print('Word: World')
    world_clf = train_clf(world_df, world_df[:5])
    print('Word: Deal')
    deal_clf = train_clf(deal_df, deal_df[:5])
    
    dev_df, test_df = load_dev_test()
    
    print('-----------------------')
    
    print('Evaluate on dev set')
    print('Word: Country')
    test_clf(country_clf, dev_df)
    print('Word: Action')
    test_clf(action_clf, dev_df)
    print('Word: Week')
    test_clf(week_clf, dev_df)
    print('Word: World')
    test_clf(world_clf, dev_df)
    print('Word: Deal')
    test_clf(deal_clf, dev_df)
    
    print('-----------------------')
    
    print('Evaluate on test set')
    print('Word: Country')
    test_clf(country_clf, test_df)
    print('Word: Action')
    test_clf(action_clf, test_df)
    print('Word: Week')
    test_clf(week_clf, test_df)
    print('Word: World')
    test_clf(world_clf, test_df)
    print('Word: Deal')
    test_clf(deal_clf, test_df)
    


if __name__ == "__main__":
    main()




