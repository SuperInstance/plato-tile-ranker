"""Multi-signal tile ranking."""

from typing import Optional

class TileRanker:
    SIGNALS = {
        "keyword": 0.30, "confidence": 0.20, "temporal": 0.15,
        "frequency": 0.10, "ghost": 0.15, "controversy": 0.10,
    }

    def __init__(self, weights: dict = None, keyword_gate: float = 0.01):
        self.weights = weights or dict(self.SIGNALS)
        self.keyword_gate = keyword_gate

    def _keyword_overlap(self, query: str, content: str) -> float:
        if not query: return 0.0
        q = set(query.lower().split())
        c = set(content.lower().split())
        if not q or not c: return 0.0
        return len(q & c) / len(q | c)

    def score_tile(self, tile: dict, query: str = "") -> float:
        content = tile.get("content", "")
        kw = self._keyword_overlap(query, content)
        if kw < self.keyword_gate:
            return 0.0

        conf = tile.get("confidence", 0.5)
        use_count = tile.get("use_count", 0)
        freq = min(use_count / max(use_count + 1, 1), 1.0)
        temporal = 1.0 if not tile.get("expired") else 0.5
        ghost = 1.0 if not tile.get("is_ghost") else 0.3
        controversy = tile.get("controversy_bonus", 0.0)

        return (kw * self.weights["keyword"] + conf * self.weights["confidence"] +
                temporal * self.weights["temporal"] + freq * self.weights["frequency"] +
                ghost * self.weights["ghost"] + controversy * self.weights["controversy"])

    def rank(self, tiles: list[dict], query: str = "", top_n: int = 10) -> list[dict]:
        scored = []
        for t in tiles:
            s = self.score_tile(t, query)
            priority = t.get("priority", "P2")
            if priority == "P0": s += 10.0
            elif priority == "P1": s += 1.0
            t["_score"] = s
            scored.append(t)
        scored.sort(key=lambda x: -x.get("_score", 0))
        return scored[:top_n]

    def update_weights(self, weights: dict):
        self.weights.update(weights)

    @property
    def current_weights(self) -> dict:
        return dict(self.weights)
