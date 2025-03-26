import re
import sys
import unicodedata
import xxhash
import numpy as np
from pathlib import Path
from collections import defaultdict
from py_rust_stemmers import SnowballStemmer
from typing import List, Tuple, Set, Dict


def get_all_punctuation() -> Set[str]:
    return set(chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P"))

def remove_non_alphanumeric(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)


class SimpleTokenizer:
    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = re.sub(r"[^\w]", " ", text.lower())
        text = re.sub(r"\s+", " ", text)
        return text.strip().split()

def load_stopwords(model_dir: Path, language: str) -> Set[str]:
    stopwords_path = model_dir / "EN-Stopwords1.txt"
    if not stopwords_path.exists():
        print("Stopwords file not found. Using empty set.")
        return set()
    with stopwords_path.open("r", encoding="utf-8") as f:
        stopwords = set(line.strip() for line in f if line.strip())
  
    return stopwords

def process_sentence(sentence: str,
                     language: str = "english",
                     token_max_length: int = 40,
                     disable_stemmer: bool = False,
                     stopwords_dir: Path = None) -> List[str]:
    if stopwords_dir is not None:
        print("Stopwords file provided, attempting to load...")
        stopwords = load_stopwords(stopwords_dir, language)
    else:
        print("No stopwords directory provided. Using default stopwords.")
        stopwords = {"the", "is", "at", "which", "on", "a", "an", "and", "or", "of", "for", "in", "to"}
    
    punctuation = get_all_punctuation()
    stemmer = SnowballStemmer(language) if not disable_stemmer else None

    cleaned = remove_non_alphanumeric(sentence)
    tokens = SimpleTokenizer.tokenize(cleaned)

    processed_tokens = []
    for token in tokens:
        lower_token = token.lower()
        if token in punctuation:
            continue
        if lower_token in stopwords:
            continue
        if len(token) > token_max_length:
            continue
        stemmed_token = stemmer.stem_word(lower_token) if stemmer else lower_token
        if stemmed_token:
            processed_tokens.append(stemmed_token)
    return processed_tokens

def count_and_clamp_frequencies(tokens: List[str]) -> Dict[str, int]:
    freq = defaultdict(int)
    for token in tokens:
        freq[token] += 1
    return {token: (count if count <= 8 else 8) for token, count in freq.items()}

def hash_token(token: str) -> int:
    return xxhash.xxh32(token.encode("utf-8")).intdigest() & 0xFFFFFFFF

def construct_sparse_vector(tokens: List[str]) -> Tuple[List[Tuple[int, np.uint8]], np.uint16]:
    freq = count_and_clamp_frequencies(tokens)
    sparse_vector = [(hash_token(token), np.uint8(count)) for token, count in freq.items()]
    doc_length = np.uint16(len(tokens))
    return sparse_vector, doc_length

# --- Main Pipeline Example ---

if __name__ == "__main__":
    document = (
        "This is an example document for BM25 sparse vector creation. "
        "It demonstrates tokenization, stopword removal, stemming, "
        "frequency clamping, and hashing using XXHash32. " * 10
    )
    
    tokens = process_sentence(document, language="english", stopwords_dir=Path("."))
    
    sparse_vector, doc_length = construct_sparse_vector(tokens)
    
    print("Sparse Vector:", sparse_vector)
    print("Document Length:", doc_length)
    print("Document Length Type:", type(doc_length))
