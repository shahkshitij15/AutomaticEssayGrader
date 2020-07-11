import os
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential, load_model, model_from_config
import keras.backend as K
from keras.optimizers import SGD, Adam

# X = pd.read_csv(('training_set_rel3.tsv'), sep='\t', encoding='ISO-8859-1')
# y = X['domain1_score']
# X = X.dropna(axis=1)
# X = X.drop(columns=['rater1_domain1', 'rater2_domain1'])

nltk.download('punkt')
nltk.download('stopwords')


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


# X_set_1 = X[X['essay_set'] == 1]
# X_set_2 = X[X['essay_set'] == 2]
# X_set_3 = X[X['essay_set'] == 3]
# X_set_4 = X[X['essay_set'] == 4]
# X_set_5 = X[X['essay_set'] == 5]
# X_set_6 = X[X['essay_set'] == 6]
# X_set_7 = X[X['essay_set'] == 7]
# #X_set_8 = X[X['essay_set'] == 8]
# y_set_1 = y[X['essay_set'] == 1]
# y_set_2 = y[X['essay_set'] == 2]
# y_set_3 = y[X['essay_set'] == 3]
# y_set_4 = y[X['essay_set'] == 4]
# y_set_5 = y[X['essay_set'] == 5]
# y_set_6 = y[X['essay_set'] == 6]
# y_set_7 = y[X['essay_set'] == 7]
# #y_set_8 = y[X['essay_set'] == 8]

from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import cohen_kappa_score

#cv = KFold(n_splits=5, shuffle=True)
def fit_model(X_essay, y_score, num):
    results = []
    y_pred_list = []

    count = 1
    #for traincv, testcv in cv.split(X):
    #   print("\n--------Fold {}--------\n".format(count))
    X_train, X_test, y_train, y_test = train_test_split(X_essay, y_score, test_size = 0.2, random_state = 26 )
        
    training_essay = X_train['essay']
    testing_essay = X_test['essay']
        
    sentences = []
        
    for essay in training_essay:
                # Obtaining all sentences from the training essays.
        sentences += essay_sentence(essay, rm_stopwords = True)
                
        # Initializing variables for word2vec model.
    num_feat = 300
    epoch = 100
    batch = 40
    print("Training Word2Vec Model...")
    model = Word2Vec(sentences, workers=5, size= 300 , min_count = 40, window = 10, sample = (1e-3))

    model.init_sims(replace=True)
    model.wv.save_word2vec_format('./AES_weights/word2vecmodel_'+num+'.bin', binary=True)

    clean_train_essays = []
    for e in training_essay:
        clean_train_essays.append(essay_words(e, rm_stopwords=True))
    train_data = Avg_feat_vect(clean_train_essays, model, num_feat)
        
    clean_test_essays = []
    for e in testing_essay:
        clean_test_essays.append(essay_words( e, rm_stopwords=True ))
    test_data = Avg_feat_vect( clean_test_essays, model, num_feat )
        
    train_data = np.array(train_data)
    test_data = np.array(test_data)
        # Reshaping train and test vectors to 3 dimensions. (1 represnts one timestep)
    train_data = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1]))
    test_data = np.reshape(test_data, (test_data.shape[0], 1, test_data.shape[1]))
        
    lstm_model = build_model()
    lstm_model.fit(train_data, y_train, batch_size=batch, epochs=epoch)
    #lstm_model.load_weights('./AES_weights/aes_weights_'+num+'.h5')
    y_pred = lstm_model.predict(test_data)
        
    #if count == 5:
    lstm_model.save('./AES_weights/aes_weights_'+num+'.h5')
        
    y_pred = np.around(y_pred)
    return y_pred,y_test,model

'''y_pred_1, y_test_1, model_1 = fit_model(X_set_1, y_set_1,'1')
y_pred_2, y_test_2, model_2 = fit_model(X_set_2, y_set_2,'2')
y_pred_3, y_test_3, model_3 = fit_model(X_set_3, y_set_3,'3')
y_pred_4, y_test_4, model_4 = fit_model(X_set_4, y_set_4,'4')
y_pred_5, y_test_5, model_5 = fit_model(X_set_5, y_set_5,'5')
y_pred_6, y_test_6, model_6 = fit_model(X_set_6, y_set_6,'6')
y_pred_7, y_test_7, model_7 = fit_model(X_set_7, y_set_7,'7')
#y_pred_8, y_test_8, model_8 = fit_model(X_set_8, y_set_8,'8')'''


def driver(add_new_topic):
    y_new_essay = add_new_topic['domain1_score']
    add_new_topic = add_new_topic.dropna(axis=1)
    add_new_topic = add_new_topic.drop(columns=['rater1_domain1', 'rater2_domain1'])
    X_set_8 = add_new_topic[add_new_topic['essay_set'] == 8]
    y_set_8 = y_new_essay[add_new_topic['essay_set'] == 8]
    y_pred_8, y_test_8, model_8 = fit_model(X_set_8, y_set_8,'8')

