import os

def load_documents(folder):

    docs = []

    for file in os.listdir(folder):

        if file.endswith(".txt"):

            path = os.path.join(folder, file)

            with open(path, "r") as f:
                docs.append(f.read())

    return docs


def chunk_text(text, chunk_size=300):

    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    return chunks