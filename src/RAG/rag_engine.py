import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGStore:
    def __init__(self, embed_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embed_model)
        self.index = None
        self.texts = []
        self.metas = []
        self.df = None
        self.data_dir = "./data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.faiss_path = os.path.join(self.data_dir, "faiss.index")
        self.meta_path = os.path.join(self.data_dir, "faiss_meta.pkl")
        self.embeddings_path = os.path.join(self.data_dir, "embeddings.npy")

    def build_index(self, df):
        self.df = df
        docs, metas = [], []
        for i, r in df.iterrows():
            text = f"{r['property_id']} | {r['address']} | {r['borough']} | {r['bedrooms']} bed | £{r['price']}"
            docs.append(text)
            metas.append({"pid": r["property_id"], "row": i})
        emb = self.model.encode(docs, convert_to_numpy=True, show_progress_bar=True)
        np.save(self.embeddings_path, emb)
        idx = faiss.IndexFlatL2(emb.shape[1])
        idx.add(emb.astype(np.float32))
        self.index, self.texts, self.metas = idx, docs, metas
        faiss.write_index(idx, self.faiss_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump({"texts": docs, "metas": metas}, f)
        return {"count": len(docs)}

    def load_index(self):
        if os.path.exists(self.faiss_path):
            self.index = faiss.read_index(self.faiss_path)
            with open(self.meta_path, "rb") as f:
                meta = pickle.load(f)
                self.texts, self.metas = meta["texts"], meta["metas"]
            return True
        return False

    def query(self, q, k=5):
        if not self.index:
            raise ValueError("FAISS index not built or loaded.")
        q_emb = self.model.encode([q], convert_to_numpy=True)
        D, I = self.index.search(q_emb.astype(np.float32), k)
        results = []
        for d, idx in zip(D[0], I[0]):
            meta = self.metas[idx]
            results.append({
                "property_id": meta["pid"],
                "distance": float(d),
                "snippet": self.texts[idx]
            })
        return results
