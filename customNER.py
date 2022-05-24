# import important modules
import numpy as np
import pandas as pd
# sklearn modules
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plot
import spacy
from spacy.lang.en import English

import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training import Example

import nltk
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import logging


logger = logging.getLogger(__name__)


nlppipeline = spacy.load('en_core_web_sm')

# Getting the pipeline component
ner = nlppipeline.get_pipe("ner")
nlp = spacy.load("en_core_web_sm")

optimizer = nlp.resume_training()

# Download dependency
for dependency in (
        "brown",
        "names",
        "wordnet",
        "averaged_perceptron_tagger",
        "universal_tagset",
):
    nltk.download(dependency)

import warnings

warnings.filterwarnings("ignore")
# seeding
np.random.seed(123)





###########################Load data

flight_details = pd.read_csv('atis_data.csv',encoding='utf-8')

df = pd.DataFrame(flight_details)


print(flight_details.head())
############ data visualization ###########
def data_visualization(df):
    fig, graph = plot.subplots()
    p1 = graph.bar(df['Intents'], df['Input_Queries'], align='edge', width=0.3)
    graph.yaxis.set_visible(False)
    plot.show()
    return df
#######################data cleanup
def data_preprocessing(df):
    input_convos = df['Input_Queries']

    # Removing punctuation
    df['final_processed_data'] = input_convos.map(lambda x: re.sub('[,\.!?]', '', x))

    # Converting the dataset to lowercase
    df['final_processed_data'] = input_convos.map(lambda x: x.lower())
    lemmatizer = WordNetLemmatizer()
    final_processed_data = []
    for data in df['final_processed_data']:
        lemmatized_data = lemmatizer.lemmatize(data)
        final_processed_data.append(lemmatized_data)

    df['final_processed_data'] = final_processed_data
    #df.head()
    return df
def data_train_test_split(df):
    X=df['Input_Queries']
    y=df['Intents']
    X_train, X_test, y_train, y_test=train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        shuffle=True,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def nlp_ner(df):
    query = nlp(df)
    ners = {}
    for word in query.ents:
        if word.label_ != 'GPE':
            ners[word.label_] = word.text
    return ners
        #displacy.render(query, style="ent", jupyter=True)

def rule_based_ner(df):
    data = df['final_processed_data']
    sourceloc = ''
    destloc = ''
    destination = []
    source = []
    source_data = []
    destination_data = []
    for doc in data:
        words_list = doc.split()

        if ' from ' in doc:
            sourceloc = words_list[words_list.index('from') + 1]
        elif ' leaving ' in doc:
            sourceloc = words_list[words_list.index('leaving') + 1]
        else:
            doc = doc + ' na'
            sourceloc = 'na'
        nlpner = English()
        ruler = nlpner.add_pipe("entity_ruler")
        source_rules = [{"label": "source", "pattern": [{"LOWER": sourceloc}]}]
        sourceloc = ''
        ruler.add_patterns(source_rules)

        if ' in ' in doc:
            destloc = words_list[words_list.index('in') + 1]
        elif ' to ' in doc:
            destloc = words_list[words_list.index('to') + 1]
        else:
            destloc = 'na'
            doc = doc + ' na'
        dest_rules = [{"label": "destination", "pattern": [{"LOWER": destloc}]}]
        ruler.add_patterns(dest_rules)
        destloc = ''
        sourceloc = ''
        doc1 = nlpner(doc)
        for entity in doc1.ents:
            if entity.label_ == 'source':
                source.append(entity.text)
                source_data.append((doc, {'entities': [(doc.index(entity.text),
                                                        doc.index(entity.text) + len(entity.text),
                                                        'SOURCE_LOC')]}))
                break
        for entity in doc1.ents:
            if entity.label_ == 'destination':
                destination.append(entity.text)
                destination_data.append((doc, {'entities': [(doc.index(entity.text),
                                                             doc.index(entity.text) + len(
                                                                 entity.text),
                                                             'DESTINATION_LOC')]}))
                break


    df['source'] = source
    df['destination'] = destination
    print(df.head())
    return source_data,destination_data

def ner_train_test_split(source_data,destination_data):
    n = len(source_data)
    print('Total data length: ', n)
    train_data_size = n * 0.7
    test_data_size = n * 0.3
    source_train_data = source_data[0:int(train_data_size)]
    source_test_data = source_data[int(train_data_size):]
    destination_train_data = destination_data[0:int(train_data_size)]
    destination_test_data = destination_data[int(train_data_size):]
    print('source split: ', len(source_train_data), len(source_test_data))
    print('destination split: ', len(destination_train_data), len(destination_test_data))
    return source_train_data,source_test_data,destination_train_data,destination_test_data


def custom_modelling(label, traindata):

    output_path = ''
    for _, annotates in traindata:
        for ent in annotates.get("entities"):
            ner.add_label(ent[2])
    # Disabling the components otherthan the required ones
    unaffected_pipelines = [pipeline for pipeline in nlppipeline.pipe_names if
                            pipeline not in ["ner", "trf_wordpiecer", "trf_tok2vec"]]

    # Model Training with 40 iterations so that it wont remember the data
    with nlppipeline.disable_pipes(*unaffected_pipelines):

        for iteration in range(30):
            random.shuffle(traindata)
            losses = {}
            #  using spaCy's minibatch to batch up the train data
            allbatches = minibatch(traindata, size=compounding(5.0, 30.0, 1.001))
            for eachbatch in allbatches:
                for txt, annotates in eachbatch:
                    doc = nlppipeline.make_doc(txt)
                    example = Example.from_dict(doc, annotates)
                    # Running nlppipeline.update to adjust the weights
                    nlppipeline.update([example], losses=losses, drop=0.3)
                    # print(losses)

    # Saving the model to path same as the label so that it can be loaded from the same path again

    output_path = Path(label)
    print("Saving the model to", output_path)
    nlppipeline.to_disk(output_path)


def custom_spacy_ner(source_train_data,source_test_data,destination_train_data,destination_test_data):

    custom_modelling('source', source_train_data)
    custom_modelling('destination', destination_train_data)


# data_preprocessing(df)
# source_data, destination_data = rule_based_ner(df)
# source_train_data, source_test_data, destination_train_data, destination_test_data = ner_train_test_split(    source_data, destination_data)
# custom_spacy_ner(source_train_data, source_test_data, destination_train_data, destination_test_data)

#data_visualization(df)

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def get_ner(query):
    logger.info('entered get_ner')
    source = ''
    dest = ''
    all_entities=nlp_ner(query)
    for output_dir in ['source','destination']:
        print("Loading from", output_dir)
        move_names = list(ner.move_names)
        nlp2 = spacy.load(output_dir)
        #assert nlp2.get_pipe("ner").move_names == move_names
        print(query)
        doc2 = nlp2(query)

        for ent in doc2.ents:
            print(ent.label_, ent.text)
            if ent.label_=='SOURCE_LOC':
                obtained_source=ent.text
                source= obtained_source
            if ent.label_=='DESTINATION_LOC':
                obtained_dest=ent.text
                dest=obtained_dest

    location_entities={'source':source,'dest':dest}
    # print(location_entities)
    # print(all_entities)
    #result = all_entities | location_entities
    result_entities =Merge(all_entities, location_entities)
    result = {'Entities': result_entities}
    logger.info("All the required entities have been fetched: "+str(result))
    print(result)
    return result
