from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def tf_idf_calculator(data):
    tf_idf = TfidfVectorizer(sublinear_tf=True, min_df=2, norm='l2', encoding='latin-1', ngram_range=(1, 3),
                             stop_words='english')
    transformed = tf_idf.fit_transform(data)
    index_value = {i[1]: i[0] for i in tf_idf.vocabulary_.items()}

    fully_indexed = []
    transformed = np.array(transformed.todense())
    for row in transformed:
        fully_indexed.append({index_value[column]: value for (column, value) in enumerate(row)})

    return fully_indexed
