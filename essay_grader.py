import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential, load_model, model_from_config
import keras.backend as K
from keras.optimizers import SGD, Adam
import gensim
import pickle

nltk.download('punkt')
nltk.download('stopwords')

minimum_scores = [-1, 2, 1, 0, 0, 0, 0, 0, 0]
maximum_scores = [-1, 12, 6, 3, 3, 4, 4, 30, 60]

def essay_words(essay, rm_stopwords):
    """Remove the tagged labels and word tokenize the sentence."""
    essay = re.sub("[^a-zA-Z]", " ", essay)
    words = essay.lower().split()
    if rm_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)

def essay_sentence(essay, rm_stopwords):
    """Sentence tokenize the essay and call essay_words() for word tokenization."""
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    token_sentences = tokenizer.tokenize(essay.strip())
    sentences = []
    for s in token_sentences:
        if len(s) > 0:
            sentences.append(essay_words(s, rm_stopwords))
    return sentences

def feat_vect(words, model, num_feat):
    """Make Feature Vector from the words list of an Essay."""
    featureVec = np.zeros((num_feat,),dtype="float32")
    num_words = 0.
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            num_words += 1
            featureVec = np.add(featureVec,model[word])        
    featureVec = np.divide(featureVec,num_words)
    return featureVec

def Avg_feat_vect(essays, model, num_feat):
    """Main function to generate the word vectors for word2vec model."""
    c = 0
    essay_feat = np.zeros((len(essays),num_feat),dtype="float32")
    for e in essays:
        essay_feat[c] = feat_vect(e, model, num_feat)
        c = c + 1
    return essay_feat

def build_model():
    """Define the model."""
    model = Sequential()
    model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2, input_shape=[1,300], return_sequences=True))
    model.add(LSTM(100, recurrent_dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation = 'relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    return model

def conversion(score, e_set):
  e_set = int(e_set)
  final_score = int((25*score)/maximum_scores[e_set])
  return final_score

def pred_score(essay_set, text):
    clean_text = []
    
    #for t in text:
    
    clean_text.append(essay_words(text, rm_stopwords=True))
    model_w2v = gensim.models.KeyedVectors.load_word2vec_format('./AES_weights/word2vecmodel_'+essay_set+'.bin', binary=True)
    pred_text = Avg_feat_vect(clean_text, model_w2v, 300)

    pred_text = np.array(pred_text)
    pred_text = np.reshape(pred_text,(pred_text.shape[0], 1, pred_text.shape[1]))
    pred_model = build_model()

    pred_model.load_weights('./AES_weights/aes_weights_'+essay_set+'.h5')
    score = pred_model.predict(pred_text)
    score = np.around(score)
    #score = score[~(np.isnan(score))]
    final_score = conversion(score, essay_set)
    return final_score

def grade_score(essay_set,scoring_essay):
    grade = pred_score(essay_set, scoring_essay)
    return grade

