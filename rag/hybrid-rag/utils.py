import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def keyword_search(query, documents):

    results = []

    for doc in documents:
        if query.lower() in doc.lower():
            results.append(doc)

    return results