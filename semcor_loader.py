#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk.corpus import semcor
from nltk.tree import Tree
from nltk.corpus import stopwords
import pickle
import string
import numpy as np
from pandas import DataFrame


# In[2]:


class custom_WSDInstance:
    def __init__(self, lemma, context,gold_key):
        self.lemma = lemma      # lemma of the word whose sense is to be resolved
        self.context = context  # lemma of all the words in the sentential context
        self.gold_key = gold_key
    def __str__(self):
        context = ' '.join(self.context)
        return '%s\n%s\n%s\n' % ( self.lemma, context, self.gold_key)

def flatten_context(context):
        new_context = []
        for entry in context:
            if type(entry) is list:
                for word in entry:
                    new_context.append(word)
            else:
                new_context.append(entry)
        return new_context
        
def get_context(sentence):
    context = []
    for i, el in enumerate(sentence):
        if type(el) is list:
            context.append(el)
        else:
            # type(el) is Tree
            lexical_term = el.leaves()
            if type(lexical_term) is list:
                context.append(lexical_term)
    context = flatten_context(context)
    return context


# In[3]:


def load(word):
    train_instances = []
    
    for sentence in semcor.tagged_sents(tag='sem')[:]:
        context = get_context(sentence)
        for el in sentence:
            if type(el) is Tree:
                # type(el) is Tree
                lemm = ' '.join(el.leaves())   
                
                if word != None and lemm != word:
                    continue
                
                try: 
                    golden_key = el.label().key()
                except AttributeError:
                    continue
                one_instance = custom_WSDInstance(lemm, context, golden_key)
                train_instances.append(one_instance)
    
    return train_instances


# In[4]:


def dump_to_disk(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
def retriv_from_disk(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


# In[5]:


def load_five_term():
    country = load('country')
    f1 = "country_%s.pkl" % len(country)
    dump_to_disk(country, f1)
    print('Finish country')
    
    action = load('action')
    f2 = "action_%s.pkl" % len(action)
    dump_to_disk(action, f2)
    print('Finish action')
    
    week = load('week')
    f3 = "week_%s.pkl" % len(week)
    dump_to_disk(week, f3)
    print('Finish week')
    
    world = load('world')
    f4 = "world_%s.pkl" % len(world)
    dump_to_disk(world, f4)
    print('Finish world')
    
    deal = load('deal')
    f5 = "deal_%s.pkl" % len(deal)
    dump_to_disk(deal, f5)
    print('Finish deal')
    
    return country, action, year, friday, deal


# In[6]:


def main():
    load_five_term()


# In[7]:


if __name__ == "__main__":
    main()

