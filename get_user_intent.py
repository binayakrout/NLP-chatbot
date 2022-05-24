import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, Dense, Activation, Flatten, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

#function to load the dataset
def intent_data_load():
 atis_trainData = pd.read_csv('atis_intents_train.csv')
 atis_testData = pd.read_csv('atis_intents_test.csv')
 col=atis_trainData.columns
 df_length = len(atis_trainData)
 atis_trainData.loc[df_length] = col
 #rename the column
 atis_trainData.columns = ['Intent','Sentence']
 	
 col=atis_testData.columns
 df_length = len(atis_testData)
 atis_testData.loc[df_length] = col
 #rename the column
 atis_testData.columns = ['Intent','Sentence']
 combine = [atis_trainData, atis_testData]

 #combine the dataset
 flightData = pd.concat(combine)
 intents=flightData.iloc[:,0]
 #intent column as list
 intents=list(intents)
 sentence=flightData.iloc[:,1]
 #sentence column as list
 sentence=list(sentence)
 return intents,sentence

#function for one hot coding
def yonehot(intents):
	laEn = LabelEncoder()
	trans_intent = laEn.fit_transform(intents)              #convert intents into nos
	trans_intent = laEn.transform(intents)
	onehotintent = to_categorical(np.asarray(trans_intent)) #convert the nos into one hot coding
	return onehotintent

#function to split the dataset
def datasplit(intents,sentence):
	#split the dataset into 70% train and 30% valiation
	size = int(len(sentence)*0.3)
	xtrain = sentence[size:] 
	ytrain = intents[size:] 
	xvalidation = sentence[:size] 
	yvalidation = intents[:size]
	return xtrain,ytrain,xvalidation,yvalidation

#function to preprocess the sentence
def intent_preprocessing(inputstr):
	intents,sentence = intent_data_load()
	# text preprocessing
	token = Tokenizer(num_words=5000)
	token.fit_on_texts(sentence)
	#tokenize the sentences
	seq_inputstr = token.texts_to_sequences(inputstr)

	#padding the tokenized sentences
	pad_inputstr = sequence.pad_sequences(seq_inputstr, maxlen = 20)
	return pad_inputstr

#train the model
def modeltrain(xtrain,ytrain,xvalidation,yvalidation):
    #model initialization 
	rnn_mod = Sequential()
	rnn_mod.add(Embedding(5000, 60, input_length = 20)) 
	rnn_mod.add(LSTM(200))
	#addition of layers
	rnn_mod.add(Dense(1000, activation='relu'))
	#addition of layers
	rnn_mod.add(Dense(600, activation='relu'))
	#addition of layers
	rnn_mod.add(Dense(8, activation='softmax'))
	rnn_mod.summary()
	optimum = tf.keras.optimizers.Nadam(learning_rate=0.01, beta_1=0.8, beta_2=0.8,schedule_decay=0.002, epsilon=1e-08)     #parameter setting
	rnn_mod.compile(loss='categorical_crossentropy', optimizer=optimum, metrics=['accuracy'])                               #model compile
	rnn_mod.fit(xtrain,ytrain, epochs = 10, batch_size=50, verbose=1, validation_data=(xvalidation, yvalidation))           #model training
    
	return rnn_mod

#to train and store the model
def trainandstore():
	intents,sentence = intent_data_load()
	intents = yonehot(intents)
	xtrain,ytrain,xvalidation,yvalidation = datasplit(intents,sentence)
	xtrain = intent_preprocessing(xtrain)
	xvalidation = intent_preprocessing(xvalidation)
	trainedmodel = modeltrain(xtrain,ytrain,xvalidation,yvalidation)
	#store the trained model
	trainedmodel.save("trained_intent_model.h5")

#to predict the user intent

def get_intent(inuptstr):
	logger.info("Getting the intent model for prediction")
	intent_model = keras.models.load_model("trained_intent_model.h5")
	processedinput = intent_preprocessing([inuptstr])
	predictedintent = intent_model.predict(processedinput)
	predictedintent=predictedintent.argmax(axis=-1)
	label=['atis_abbreviation', 'atis_aircraft', 'atis_airfare', 'atis_airline', 'atis_flight', 'atis_flight_time', 'atis_ground_service', 'atis_quantity']
	predictedintent_labeled=label[predictedintent[0]]
	result={'userinput':inuptstr,'intent':predictedintent_labeled}
	logger.info("End of Intent prediction")
	print(result)
	return result

# x = get_intent("what is the arrival time in sanfrancisco for the 755 am flight leaving washington")
# print(x)