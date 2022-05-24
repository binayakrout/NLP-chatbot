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