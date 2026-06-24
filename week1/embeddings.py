import numpy as np


EMBEDDING_DIM = 384


def random_embedding(dim: int) -> np.ndarray:
    """Generate a random embedding vector of given dimension."""
    return np.random.rand(dim)


def random_embeddings(num_embeddings: int, dim: int) -> np.ndarray:
    """Generate a matrix of random embedding vectors."""
    return np.random.rand(num_embeddings, dim)


def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def embed_text(text: str) -> np.ndarray:
    """Create a fake embedding for a single text."""
    return random_embedding(EMBEDDING_DIM)


def embed_texts(texts) -> np.ndarray:
    """Create fake embeddings for multiple texts."""
    return random_embeddings(len(texts), EMBEDDING_DIM)


if __name__ == "__main__":
    # Example usage
    dim = 5
    num_embeddings = 10

    embedding = random_embedding(dim)
    print("Random Embedding:", embedding)

    embeddings_matrix = random_embeddings(num_embeddings, dim)
    print("Random Embeddings Matrix:\n", embeddings_matrix)

    normalized_vector = normalize(embedding)
    print("Normalized Embedding:", normalized_vector)