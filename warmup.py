"""Download models ahead of the demo so the first UI load is instant."""
import time

print("[1/2] Downloading sentence-transformers/all-MiniLM-L6-v2 ...")
t0 = time.time()
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
print(f"      done in {time.time()-t0:.1f}s")

print("[2/2] Downloading cross-encoder/nli-deberta-v3-base ...")
t0 = time.time()
from transformers import AutoTokenizer, AutoModelForSequenceClassification
AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-base")
AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-base")
print(f"      done in {time.time()-t0:.1f}s")

print("All models cached. Ready for the demo.")
