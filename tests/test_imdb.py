import string
import pandas as pd
import numpy as np
from random import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('../datasets/imdb/labeledTrainData.tsv', sep='\t', header=0)

text = df['review'].values
y = df['sentiment'].values

# print(text[:10])
# print(y[:10])

TABLE_TRANS = str.maketrans({key: ' ' for key in string.punctuation})
TABLE_TRANS['.'] = ' . '
TABLE_TRANS['?'] = ' . '
TABLE_TRANS['!'] = ' . '
TABLE_TRANS[':'] = ' . '
TABLE_TRANS[';'] = ' . '
TABLE_TRANS['('] = ' '
TABLE_TRANS[')'] = ' '
TABLE_TRANS['['] = ' '
TABLE_TRANS[']'] = ' '
TABLE_TRANS[','] = ' '
TABLE_TRANS['{'] = ' '
TABLE_TRANS['}'] = ' '


def clean_text(s, first_words=0):
    # transforms sentence for word processing into a list a words
    words = s.lower().translate(TABLE_TRANS).split()
    if first_words != 0:
        words = words[:first_words]
    return " ".join(words)


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


mode = 'w2v'
if mode == 'bow':
    # --------------------------------------------------------------------
    # Bag of words
    text = [clean_text(s) for s in text]
    vectorizer = CountVectorizer(max_features=500, ngram_range=(1, 2))
    X = vectorizer.fit_transform(text).todense()
    text_len = np.array([len(s) for s in text]).reshape(len(text), 1)
    print('len text', len(text))
    print(type(X))
    print(np.shape(X))
    X = np.concatenate((X, text_len), axis=1)
    dft = pd.DataFrame(vectorizer.transform(text).todense())
    print('len dft', len(dft))

    print(len(vectorizer.get_feature_names()))
    print(vectorizer.get_feature_names()[:10])
    print(['word_'+x.replace(' ', '_') for x in vectorizer.get_feature_names()][:10])

elif mode == 'w2v':
    # --------------------------------------------------------------------
    # Word2Vec
    print('generating word2vec')
    text = [clean_text(s).split() for s in text]
    dim = 200
    model = Word2Vec(size=dim, iter=50)
    model.build_vocab(text)
    model.train(text, total_examples=model.corpus_count, epochs=model.iter)
    """
    train_text = text.copy()
    for i in range(10):

        print('epoch', i)
        # shuffle(train_text)
        model.train(train_text, total_examples=model.corpus_count, epochs=model.iter)
    """
    # then calculate word vector per paragraph
    print('generating paragraph vectors')
    v = []
    for s in text:
        ww = np.zeros((dim))
        n = 0
        for k, w in enumerate(s):
            if w in model.wv:
                ww += model.wv[w]
                n += 1
        if n > 0:
            v.append(ww / n)
        else:
            v.append(ww)

    # create vector
    v = np.array(v)

    print(np.shape(v))
    text_len = np.array([len(s) for s in text]).reshape(len(text), 1)
    X = np.concatenate((text_len, v), axis=1)

    print(np.shape(X))
else:
    # Doc2Vec
    print('generating doc2vec')
    train_text = [TaggedDocument(words=clean_text(s).split(), tags=[i]) for i, s in enumerate(text)]
    for i in range(10):
        print(train_text[i])

    model = Doc2Vec(min_count=5, window=10, size=100, sample=1e-4, negative=5, workers=8, iter=5)
    model.build_vocab(train_text)
    # train_text = text.copy()

    for i in range(2):
        print('epoch', i)
        shuffle(train_text)
        model.train(train_text, total_examples=model.corpus_count, epochs=model.iter)

    # create vector
    v = model.docvecs[list(range(len(text)))]
    print(v[:5, :4])
    print(np.shape(v))
    print('infer vector')
    v = np.array([model.infer_vector(clean_text(s).split(), steps=10) for s in text])
    print(v[:5, :4])
    print(np.shape(v))
    text_len = np.array([len(s) for s in text]).reshape(len(text), 1)
    X = np.concatenate((text_len, v), axis=1)
    X = v
    print(np.shape(X))


print('training')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = LogisticRegression()
model.fit(X_train, y_train)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

y_pred = model.predict(X_test)
print('accuracy:', sum(y_pred == y_test) / len(y_test))

y_pred = model.predict_proba(X_test)
print('log loss:', log_loss(y_test, y_pred))
