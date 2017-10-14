import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from gensim.models import Word2Vec

df = pd.read_csv('../datasets/imdb/labeledTrainData.tsv', sep='\t', header=0)

text = df['review'].values
y = df['sentiment'].values

print(text[:10])
print(y[:10])

TABLE_TRANS = str.maketrans({key: ' ' for key in string.punctuation})


def clean_text(s):
    # transforms sentence for word processing into a list a words
    return s.lower().translate(TABLE_TRANS).split()


def to_array(x):
    if isinstance(x, list):
        return np.reshape(x, (len(x), 1))
    elif isinstance(x, np.ndarray):
        if len(np.shape(x)) == 1:
            return np.reshape(x, (len(x), 1))
    return x


def stack(x, y):
    if x == []:
        return to_array(y)
    else:
        return np.concatenate([x, to_array(y)], axis=1)


text = [" ".join(s.lower().translate(TABLE_TRANS).split()) for s in text]
print(text[:10])

mode = 'w2v'
if mode == 'bow':
    # --------------------------------------------------------------------
    # Bag of words
    vectorizer = CountVectorizer(max_features=500, ngram_range=(1, 2))
    X = vectorizer.fit_transform(text)
    print('len text', len(text))
    print(type(X))
    print(np.shape(X))
    dft = pd.DataFrame(vectorizer.transform(text).todense())
    print('len dft', len(dft))

    print(len(vectorizer.get_feature_names()))
    print(vectorizer.get_feature_names()[:10])
    print(['word_'+x.replace(' ', '_') for x in vectorizer.get_feature_names()][:10])
else:
    # --------------------------------------------------------------------
    # Word2Vec
    print('generating word2vec')
    dim = 200
    model = Word2Vec(text, size=dim)

    # then calculate word vector per paragraph
    print('generating paragraph vectors')
    v = []
    for s in text:
        v0 = []
        for w in s:
            if w in model.wv:
                v0 = stack(v0, model.wv[w])
        if len(v0) > 1:
            ww = np.mean(v0, axis=1)
            v.append(ww)
        else:
            v.append(np.zeros((dim)))

    v = np.array(v)
    text_len = np.array([len(s) for s in text]).reshape(len(text), 1)
    X = np.concatenate((text_len, v), axis=1)

print('training')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(sum(y_pred == y_test) / len(y_test))

y_pred = model.predict_proba(X_test)
print(log_loss(y_test, y_pred))