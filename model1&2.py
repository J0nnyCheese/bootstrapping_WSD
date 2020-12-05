#!/usr/bin/env python
# coding: utf-8


import loader as ld
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
from nltk.classify import *
from nltk.wsd import lesk
from nltk import pos_tag


def load_xml():
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = ld.load_instances(data_f)
    dev_key, test_key = ld.load_key(key_f)
    
    dev_instances = {k:v for (k,v) in dev_instances.items() if k in dev_key}
    test_instances = {k:v for (k,v) in test_instances.items() if k in test_key}
    return dev_instances, dev_key, test_instances, test_key


def run_first_common_sense(instances, key):
    predictions = []
    targets = []
    
    for id, inst in instances.items():
        lemma = inst.lemma.decode("utf-8")
        synset_key = wn.synsets(lemma)[0].lemmas()[0].key()
        predictions.append(synset_key)
        targets.append(key[id])
        
    acc = accuracy(predictions,targets)
    print('Accuracy is: ', acc)


def default_lesk(context, lemma):
    filtered_context = remove_stopwords(context)
    synset = lesk(filtered_context, lemma)
    if synset is not None:
        return synset.lemmas()[0].key()
    else:
        print('no synset for {}'.format(lemma))
        return None

def run_lesk(instances, key):
    predictions = []
    targets = []
    
    for id, inst in instances.items():
        
        lemma = inst.lemma.decode("utf-8")
        context = [el.decode("utf-8") for el in inst.context]      
        predictions.append(default_lesk(context,lemma))
        targets.append(key[id])
    
    acc = accuracy(predictions,targets)
    print('Accuracy is: ', acc)
    return acc


def accuracy(predictions, targets):
    correct = 0
    for prediction, target in zip(predictions, targets):
        if prediction in target:
            correct += 1
    return correct / len(predictions)

def remove_stopwords(context):
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []
    for w in context:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence

def main():
    dev_instances, dev_key, test_instances, test_key = load_xml()
    run_first_common_sense(dev_instances, dev_key)
    run_first_common_sense(test_instances, test_key)
    run_lesk(dev_instances, dev_key)
    run_lesk(test_instances, test_key)


if __name__ == "__main__":
    main()

