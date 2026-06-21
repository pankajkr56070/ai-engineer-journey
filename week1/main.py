"""Small demo that wires embeddings and search together."""
import os
import sys

# Ensure local imports work when running from project root
sys.path.insert(0, os.path.dirname(__file__))

import embeddings
import search


def main():
    documents = [
        "I love machine learning",
        "I love cricket",
        "Backend engineering with Go",
    ]

    # build fake embeddings (replace with real model in production)
    embs = embeddings.embed_texts(documents)

    # fake query embedding for demonstration
    query = embeddings.embed_text("I enjoy learning about models")

    results = search.nearest_documents(query, embs, documents, k=3)
    print("Top matches:")
    for score_doc in results:
        doc, score = score_doc
        print(f"{score:.4f}\t{doc}")


if __name__ == "__main__":
    main()
