"""
rag.py
--------
A lightweight Retrieval-Augmented-Generation engine.

Design choice: instead of calling an external embedding API for every
chunk (slow, costly, needs network at build time), we build a local
TF-IDF vector index with scikit-learn. This is fast, free, works
fully offline, and is more than good enough for a domain-specific
knowledge base like this one. Gemini is then only used for the final
answer *generation* step, using the retrieved chunks as context.

The index mixes three kinds of documents:
  - license procedure docs   (type="license")
  - driving school entries   (type="school")
  - maintenance price rows   (type="maintenance")

so a single semantic search can pull relevant info from any of them.
"""

from dataclasses import dataclass
from typing import List, Optional
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data_loader import parse_all_data, School, MaintenanceRow, LicenseDoc


@dataclass
class Chunk:
    doc_type: str  # "license" | "school" | "maintenance"
    text: str
    ref: dict  # original structured data, for building citations / UI cards


class RagIndex:
    def __init__(self):
        self.license_docs: List[LicenseDoc] = []
        self.schools: List[School] = []
        self.maintenance: List[MaintenanceRow] = []
        self.chunks: List[Chunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self._build()

    def _build(self):
        self.license_docs, self.schools, self.maintenance = parse_all_data()

        for d in self.license_docs:
            self.chunks.append(
                Chunk(doc_type="license", text=d.as_text(), ref=d.__dict__)
            )
        for s in self.schools:
            self.chunks.append(
                Chunk(doc_type="school", text=s.as_text(), ref=s.__dict__)
            )
        for m in self.maintenance:
            self.chunks.append(
                Chunk(doc_type="maintenance", text=m.as_text(), ref=m.__dict__)
            )

        texts = [c.text for c in self.chunks]
        # TF-IDF works fine on Arabic text as long as we don't rely on
        # English-only stopword lists. word-level n-grams (1,2) help
        # catch short multi-word phrases like "تيل الفرامل".
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 6) -> List[Chunk]:
        if not query.strip():
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        # Rank by similarity, keep only chunks with a non-zero score
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        results = [self.chunks[i] for i in ranked[:top_k] if sims[i] > 0]
        return results

    # ---- structured helpers (fast, exact filters, no ML needed) ----

    def filter_schools(self, area: str = "", governorate: str = "") -> List[School]:
        out = self.schools
        if area:
            out = [s for s in out if area.strip() in s.area]
        if governorate:
            out = [s for s in out if governorate.strip() in s.governorate]
        return out

    def filter_maintenance(
        self,
        engine_cc: str = "",
        service_type: str = "",
        city: str = "",
    ) -> List[MaintenanceRow]:
        out = self.maintenance
        if engine_cc:
            out = [m for m in out if m.engine_cc == engine_cc.strip()]
        if service_type:
            out = [m for m in out if service_type.strip() in m.service_type]
        if city:
            out = [m for m in out if city.strip() in m.city]
        return out


# Singleton index, built once when the app starts.
_index: Optional[RagIndex] = None


def get_index() -> RagIndex:
    global _index
    if _index is None:
        _index = RagIndex()
    return _index
