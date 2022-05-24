import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import re
from spacy.lang.en import English
from spacy import displacy
import logging
import spacy
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training import Example

import pandas as pd

logging.basicConfig(filename='atis_ChatBot.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

flight_details = pd.read_csv('atis_data.csv')
df = pd.DataFrame(flight_details)
flight_details.head()

nlppipeline = spacy.load('en_core_web_sm')

# Getting the pipeline component
ner = nlppipeline.get_pipe("ner")
nlp = spacy.load("en_core_web_sm")

optimizer = nlp.resume_training()

## Data pre-processing in NER

def data_preprocessing(df):
    logging.info('Preprocessing for NER')
    input_convos = df['input']

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


##NER using spacy pipeline

def nlp_ner(df):
    logging.info('Getting NER using spacy pipeline')
    query = nlp(df)
    ners = {}
    for word in query.ents:
        if word.label_ != 'GPE':
            ners[word.label_] = word.text
    logging.info('Obtained NERs from spacy pipeline', ners)
    return ners

## Model 1 Rule based entity recognizer using spacy


def rule_based_ner(df):
    logging.debug('Entering NER rule based training ')
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
    logging.debug('Completed NER rule based entity recognition')
    return source_data,destination_data


logging.info('Performing NER preprocessing and rule based')
data_preprocessing(df)
source_data, destination_data = rule_based_ner(df)


nlppipeline=spacy.load('en_core_web_sm')

# Getting the pipeline component
ner=nlppipeline.get_pipe("ner")



def ner_train_test_split(source_data,destination_data):
    logging.info('Splitting the data to train and test')
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
    logging.debug('Entering NER custom training model')

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
    logging.info("Saving the model to", output_path)
    nlppipeline.to_disk(output_path)
    logging.debug('Leaving the NER custome model')


source_train_data, source_test_data, destination_train_data, destination_test_data = ner_train_test_split(source_data, destination_data)


custom_spacy_ner(source_train_data, destination_train_data)



def get_ner(query):
    logging.debug('Entering get_ner')
    source = ''
    dest = ''
    all_entities=nlp_ner(query)
    for output_dir in ['source','destination']:
        logging.info("Loading from", output_dir)
        move_names = list(ner.move_names)
        nlp2 = spacy.load(output_dir)
        #assert nlp2.get_pipe("ner").move_names == move_names
        logging.info(query)
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
    result=all_entities | location_entities
    logging.info(result)
    return result