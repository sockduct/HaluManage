from __future__ import annotations

import re
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Optional

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - fallback path for constrained runtimes
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore
    _HAS_SKLEARN = False


@dataclass
class _Segment:
    """Represents a context fragment paired with ordering metadata."""

    text: str
    order: int
    link: Optional[str] = None
    score: float = 0.0

    @property
    def size_bytes(self) -> int:
        return len(self.text.encode("utf-8"))


class SemanticContextReducer:
    """
    Lightweight OptiLLM-style semantic reducer for large context blobs.

    The reducer splits a context string into ranked fragments using TF-IDF cosine
    similarity against the originating question. High scoring fragments are
    retained until the target byte limit is satisfied.
    """

    def __init__(
        self,
        *,
        chunk_byte_size: int = 2048,
        min_chunk_chars: int = 120,
        join_token: str = "\n\n",
    ) -> None:
        self.chunk_byte_size = max(512, chunk_byte_size)
        self.min_chunk_chars = min_chunk_chars
        self.join_token = join_token

    def reduce(
        self,
        *,
        question: str,
        context: str,
        wiki_links: Optional[Iterable[str]] = None,
        limit_bytes: int = 96 * 1024,
    ) -> str:
        """Return a compact context string that respects the provided byte limit."""
        if not context:
            return ""

        context_bytes = len(context.encode("utf-8"))
        if context_bytes <= limit_bytes:
            return context

        segments = self._build_segments(context, list(wiki_links or []))
        if not segments:
            return self._truncate_to_limit(context, limit_bytes)

        self._score_segments(question, segments)
        selected_segments = self._select_segments(segments, limit_bytes)

        if not selected_segments:
            # Fallback: take the single highest scoring segment and trim it.
            best_segment = max(segments, key=lambda seg: seg.score)
            return self._truncate_to_limit(best_segment.text, limit_bytes)

        ordered = sorted(selected_segments, key=lambda seg: seg.order)
        compact = self.join_token.join(seg.text.strip() for seg in ordered if seg.text.strip())

        if len(compact.encode("utf-8")) > limit_bytes:
            return self._truncate_to_limit(compact, limit_bytes)

        return compact

    # --------------------------------------------------------------------- Helpers

    def _build_segments(self, context: str, wiki_links: List[str]) -> List[_Segment]:
        segments: List[_Segment] = []
        order_counter = 0

        positions = []
        for link in wiki_links:
            idx = context.find(link)
            if idx != -1:
                positions.append((idx, link))
        positions.sort()

        def append_chunks(text: str, link: Optional[str], counter: int) -> int:
            for chunk in self._chunk_text(text):
                cleaned = chunk.strip()
                if not cleaned:
                    continue
                segments.append(_Segment(text=cleaned, order=counter, link=link))
                counter += 1
            return counter

        if positions:
            first_idx = positions[0][0]
            if first_idx > 0:
                prefix = context[:first_idx].strip()
                if prefix:
                    order_counter = append_chunks(prefix, None, order_counter)

            for i, (start_idx, link) in enumerate(positions):
                end_idx = positions[i + 1][0] if i + 1 < len(positions) else len(context)
                segment_text = context[start_idx:end_idx]
                order_counter = append_chunks(segment_text, link, order_counter)
        else:
            order_counter = append_chunks(context, None, order_counter)

        return segments

    def _chunk_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        sentences = self._sentence_tokenize(text)
        if not sentences:
            return [text]

        chunks: List[str] = []
        current: List[str] = []
        current_size = 0

        for sentence in sentences:
            for piece in self._split_sentence(sentence):
                piece_size = len(piece.encode("utf-8"))

                if current and current_size + piece_size > self.chunk_byte_size:
                    chunk = " ".join(current).strip()
                    if chunk:
                        chunks.append(chunk)
                    current = [piece]
                    current_size = piece_size
                else:
                    current.append(piece)
                    current_size += piece_size

                if current_size >= self.chunk_byte_size * 1.5:
                    chunk = " ".join(current).strip()
                    if chunk:
                        chunks.append(chunk)
                    current = []
                    current_size = 0

        if current:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)

        # Ensure minimum chunk length so we do not end up with lots of tiny fragments.
        merged: List[str] = []
        buffer = ""
        for chunk in chunks:
            if len(chunk) >= self.min_chunk_chars or not buffer:
                if buffer:
                    merged.append(buffer.strip())
                    buffer = ""
                merged.append(chunk)
            else:
                buffer = f"{buffer} {chunk}".strip()
        if buffer:
            merged.append(buffer.strip())

        return [chunk for chunk in merged if chunk]

    def _sentence_tokenize(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _split_sentence(self, sentence: str) -> List[str]:
        sentence = sentence.strip()
        if not sentence:
            return []

        byte_len = len(sentence.encode("utf-8"))
        if byte_len <= self.chunk_byte_size:
            return [sentence]

        words = sentence.split()
        if not words:
            return [sentence]

        pieces: List[str] = []
        current_words: List[str] = []
        current_size = 0

        for word in words:
            word_size = len(word.encode("utf-8"))
            sep = 1 if current_words else 0

            if current_words and current_size + word_size + sep > self.chunk_byte_size:
                pieces.append(" ".join(current_words).strip())
                current_words = [word]
                current_size = word_size
            else:
                if sep:
                    current_size += sep
                current_words.append(word)
                current_size += word_size

            if word_size > self.chunk_byte_size:
                pieces.append(word)
                current_words = []
                current_size = 0

        if current_words:
            pieces.append(" ".join(current_words).strip())

        return [piece for piece in pieces if piece]

    def _score_segments(self, question: str, segments: List[_Segment]) -> None:
        if _HAS_SKLEARN:
            docs = [question] + [segment.text for segment in segments]
            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform(docs)
            if matrix.shape[0] <= 1:
                for segment in segments:
                    segment.score = 0.0
                return

            question_vec = matrix[0]
            chunk_matrix = matrix[1:]
            similarities = cosine_similarity(chunk_matrix, question_vec)

            for segment, similarity in zip(segments, similarities):
                segment.score = float(similarity[0])
            return

        # Fallback TF-IDF implementation without scikit-learn
        documents = [question] + [segment.text for segment in segments]
        tokenized = [self._tokenize(doc) for doc in documents]
        doc_freq = defaultdict(int)
        for tokens in tokenized:
            for token in set(tokens):
                doc_freq[token] += 1

        def build_vector(tokens: List[str]) -> dict[str, float]:
            tf = Counter(tokens)
            vec: dict[str, float] = {}
            for term, count in tf.items():
                idf = math.log((1 + len(documents)) / (1 + doc_freq[term])) + 1
                vec[term] = (count / len(tokens)) * idf
            return vec

        vectors = [build_vector(tokens) for tokens in tokenized]
        query_vec = vectors[0]

        for segment, vec in zip(segments, vectors[1:]):
            dot = sum(query_vec.get(term, 0.0) * weight for term, weight in vec.items())
            query_norm = math.sqrt(sum(weight * weight for weight in query_vec.values())) or 1.0
            vec_norm = math.sqrt(sum(weight * weight for weight in vec.values())) or 1.0
            segment.score = dot / (query_norm * vec_norm)

    def _select_segments(self, segments: List[_Segment], limit_bytes: int) -> List[_Segment]:
        sorted_segments = sorted(segments, key=lambda seg: seg.score, reverse=True)
        selected: List[_Segment] = []
        total_bytes = 0

        for segment in sorted_segments:
            if total_bytes + segment.size_bytes <= limit_bytes:
                selected.append(segment)
                total_bytes += segment.size_bytes

        return selected

    def _truncate_to_limit(self, text: str, limit_bytes: int) -> str:
        encoded = text.encode("utf-8")
        if len(encoded) <= limit_bytes:
            return text
        truncated = encoded[:limit_bytes]
        # Attempt to avoid cutting multi-byte sequences by decoding with ignore.
        safe_text = truncated.decode("utf-8", errors="ignore")
        return safe_text.rstrip()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())
