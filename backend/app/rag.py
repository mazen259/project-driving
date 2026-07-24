"""
rag.py
--------
A stronger Retrieval-Augmented-Generation engine.

Improvements over v1:
  - Arabic-aware text normalization before indexing/searching, so
    variations like "مدرسة" vs "مدرسه", or "أ/إ/آ" vs "ا", don't
    hurt recall.
  - sublinear TF scaling + word 1-3grams for better phrase matching.
  - Hybrid boosting: if the user's question contains an exact engine
    size (e.g. "1600") or a known area/governorate/city name, chunks
    that contain that exact value get a relevance boost on top of the
    TF-IDF score. This fixes cases where a common number/place name
    has low IDF weight and would otherwise rank lower than it should.

The index still mixes three kinds of documents:
  - license procedure docs   (type="license")
  - driving school entries   (type="school")
  - maintenance price rows   (type="maintenance")
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Set

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data_loader import parse_all_data, School, MaintenanceRow, LicenseDoc

# ---------------------------------------------------------------
# Arabic normalization
# ---------------------------------------------------------------

_TASHKEEL = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u0640]")  # diacritics + tatweel


def normalize_arabic(text: str) -> str:
    """Lowercase + strip diacritics + unify letter variants so that
    semantically identical Arabic words match each other in TF-IDF."""
    if not text:
        return ""
    text = text.strip()
    text = _TASHKEEL.sub("", text)
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()


@dataclass
class Chunk:
    doc_type: str  # "license" | "school" | "maintenance"
    text: str
    norm_text: str
    ref: dict  # original structured data, for building citations / UI cards


class RagIndex:
    def __init__(self):
        self.license_docs: List[LicenseDoc] = []
        self.schools: List[School] = []
        self.maintenance: List[MaintenanceRow] = []
        self.chunks: List[Chunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self._known_ccs: Set[str] = set()
        self._known_places: Set[str] = set()
        self._build()

    def _build(self):
        self.license_docs, self.schools, self.maintenance = parse_all_data()

        for d in self.license_docs:
            txt = d.as_text()
            self.chunks.append(
                Chunk(doc_type="license", text=txt, norm_text=normalize_arabic(txt), ref=d.__dict__)
            )
        for s in self.schools:
            txt = s.as_text()
            self.chunks.append(
                Chunk(doc_type="school", text=txt, norm_text=normalize_arabic(txt), ref=s.__dict__)
            )
            self._known_places.add(normalize_arabic(s.area))
            self._known_places.add(normalize_arabic(s.governorate))
        for m in self.maintenance:
            txt = m.as_text()
            self.chunks.append(
                Chunk(doc_type="maintenance", text=txt, norm_text=normalize_arabic(txt), ref=m.__dict__)
            )
            self._known_ccs.add(m.engine_cc)
            self._known_places.add(normalize_arabic(m.city))

        texts = [c.norm_text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(texts)

    def _hybrid_boost(self, query_norm: str, chunk: Chunk, base_score: float) -> float:
        """Boost chunks that contain an exact engine size or place name
        mentioned in the query — these are strong, unambiguous signals
        that plain TF-IDF can under-weight."""
        score = base_score
        for cc in self._known_ccs:
            if cc and re.search(rf"\b{re.escape(cc)}\b", query_norm) and cc in chunk.text:
                score *= 1.35
                break
        for place in self._known_places:
            if place and len(place) > 2 and place in query_norm and place in chunk.norm_text:
                score *= 1.25
                break
        return score

    def search(self, query: str, top_k: int = 8) -> List[Chunk]:
        if not query.strip():
            return []
        query_norm = normalize_arabic(query)
        q_vec = self.vectorizer.transform([query_norm])
        sims = cosine_similarity(q_vec, self.matrix)[0]

        boosted = [
            self._hybrid_boost(query_norm, self.chunks[i], sims[i])
            for i in range(len(sims))
        ]
        ranked = sorted(range(len(boosted)), key=lambda i: boosted[i], reverse=True)
        results = [self.chunks[i] for i in ranked[:top_k] if boosted[i] > 0]
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

    def distinct_areas(self) -> List[str]:
        return sorted({s.area for s in self.schools})

    def distinct_governorates(self) -> List[str]:
        return sorted({s.governorate for s in self.schools})

    def distinct_engine_ccs(self) -> List[str]:
        return sorted({m.engine_cc for m in self.maintenance}, key=lambda x: int(x) if x.isdigit() else 0)

    def distinct_service_types(self) -> List[str]:
        return sorted({m.service_type for m in self.maintenance})

    def distinct_cities(self) -> List[str]:
        return sorted({m.city for m in self.maintenance})


# Singleton index, built once when the app starts.
_index: Optional[RagIndex] = None


def get_index() -> RagIndex:
    global _index
    if _index is None:
        _index = RagIndex()
    return _index
