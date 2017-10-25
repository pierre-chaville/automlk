import string
import numpy as np
from random import shuffle

try:
    from gensim.models import Word2Vec, Doc2Vec, fasttext
    from gensim.models.doc2vec import TaggedDocument
    import_gensim = True
except:
    import_gensim = False


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


def model_word2vec(text, params):
    # generate a word2vec model from a text (list of sentences)
    print('generating word2vec')
    train_text = [clean_text(s).split() for s in text]
    model = Word2Vec(**params)
    model.build_vocab(train_text)
    model.train(train_text, total_examples=model.corpus_count, epochs=model.iter)
    return model


def vector_word2vec(model, text, params):
    # generate an aggregate vector with words of the text
    print('generating paragraph vectors')
    v = []
    vector_text = [clean_text(s).split() for s in text]
    for s in vector_text:
        ww = np.zeros((params['size']))
        n = 0
        for k, w in enumerate(s):
            if w in model.wv:
                ww += model.wv[w]
                n += 1
        if n > 0:
            v.append(ww / n)
        else:
            v.append(ww)

    # create vector with word vectors and paragraph lenght
    v = np.array(v)
    text_len = np.array([len(s) for s in text]).reshape(len(vector_text), 1)
    return np.concatenate((text_len, v), axis=1)


def model_fasttext(text, params):
    # generate a fasttext model from a text (list of sentences)
    print('generating fasttext')
    train_text = [clean_text(s).split() for s in text]
    model = fasttext.FastText(**params)
    model.build_vocab(train_text)
    model.train(train_text, total_examples=model.corpus_count, epochs=model.iter)
    return model


def vector_fasttext(model, text, params):
    # generate an aggregate vector with words of the text
    print('generating paragraph vectors')
    v = []
    vector_text = [clean_text(s).split() for s in text]
    for s in vector_text:
        ww = np.zeros((params['size']))
        n = 0
        for k, w in enumerate(s):
            if w in model.wv:
                ww += model.wv[w]
                n += 1
        if n > 0:
            v.append(ww / n)
        else:
            v.append(ww)

    # create vector with word vectors and paragraph length
    v = np.array(v)
    text_len = np.array([len(s) for s in text]).reshape(len(vector_text), 1)
    return np.concatenate((text_len, v), axis=1)


def model_doc2vec(text, params):
    # generate a doc2vec model from a text (list of sentences)
    print('generating doc2vec')
    train_text = [TaggedDocument(words=clean_text(s).split(), tags=[i]) for i, s in enumerate(text)]
    model = Doc2Vec(**params)
    model.build_vocab(train_text)
    model.train(train_text, total_examples=model.corpus_count, epochs=model.iter)
    return model


def vector_doc2vec(model, text, params):
    # generate an a doc2vec vector from text
    print('generating paragraph vectors')
    return [model.infer_vector(clean_text(s).split()) for s in text]
