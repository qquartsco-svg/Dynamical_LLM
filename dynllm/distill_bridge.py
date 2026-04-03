"""
대도서관 지식 증류 브릿지 (DistillBridge)

대도서관(Library_of_Babel)의 LLM 응답을 DynLLM 학습 데이터로 변환한다.

흐름:
  1. 사용자 질문 → 대도서관(LLM) 호출 → 응답 획득
  2. 응답을 (input, target) 쌍으로 변환
  3. DynLLM의 온라인 적응 또는 학습 데이터셋에 축적
  4. 주기적으로 로컬 DynLLM이 이 데이터로 미세 학습

이것이 "쓰면서 배우는" 개인화 루프의 핵심:
  LLM(teacher) → 증류 → DynLLM(student) → 점점 독립적
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any


@dataclass
class DistillSample:
    """증류 데이터 단일 샘플."""
    input_text: str
    target_text: str
    source: str = "library"
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)


class DistillBuffer:
    """
    증류 데이터 버퍼.
    대도서관 응답을 축적하고, DynLLM 학습에 사용할 배치를 생성한다.
    """

    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self._samples: list[DistillSample] = []

    def add(self, sample: DistillSample):
        self._samples.append(sample)
        if len(self._samples) > self.max_samples:
            self._samples.pop(0)

    def get_training_texts(self, min_confidence: float = 0.3) -> list[str]:
        """학습에 사용할 텍스트 목록 반환 (input + target 연결)."""
        return [
            f"{s.input_text} {s.target_text}"
            for s in self._samples
            if s.confidence >= min_confidence
        ]

    def get_pairs(self, min_confidence: float = 0.3) -> list[tuple[str, str]]:
        return [
            (s.input_text, s.target_text)
            for s in self._samples
            if s.confidence >= min_confidence
        ]

    @property
    def size(self) -> int:
        return len(self._samples)

    def stats(self) -> dict:
        if not self._samples:
            return {"size": 0, "avg_confidence": 0.0}
        return {
            "size": len(self._samples),
            "avg_confidence": sum(s.confidence for s in self._samples) / len(self._samples),
            "sources": list(set(s.source for s in self._samples)),
        }

    def save(self, path: Path):
        """버퍼를 JSON lines로 저장."""
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for s in self._samples:
                f.write(json.dumps({
                    "input": s.input_text,
                    "target": s.target_text,
                    "source": s.source,
                    "confidence": s.confidence,
                    "timestamp": s.timestamp,
                }, ensure_ascii=False) + "\n")

    def load(self, path: Path):
        """저장된 버퍼 로드."""
        import json
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line.strip())
                self._samples.append(DistillSample(
                    input_text=d["input"],
                    target_text=d["target"],
                    source=d.get("source", "library"),
                    confidence=d.get("confidence", 0.5),
                    timestamp=d.get("timestamp", time.time()),
                ))


class DistillBridge:
    """
    대도서관 ↔ DynLLM 증류 브릿지.

    대도서관 커넥터를 감싸서, 모든 쿼리 결과를 자동으로 증류 버퍼에 축적한다.
    DynLLM은 이 버퍼의 데이터로 미세 학습하거나 온라인 적응할 수 있다.
    """

    def __init__(
        self,
        library_connector: Any = None,
        buffer: DistillBuffer = None,
        auto_distill: bool = True,
    ):
        self.connector = library_connector
        self.buffer = buffer or DistillBuffer()
        self.auto_distill = auto_distill
        self._query_count = 0

    def query(self, user_prompt: str, system_prompt: str = "") -> tuple[str, float]:
        """
        대도서관에 질의하고, 결과를 증류 버퍼에 축적.

        Returns:
            (응답 텍스트, 신뢰도)
        """
        if self.connector is None:
            return ("[대도서관 미연결]", 0.0)

        self._query_count += 1

        try:
            if hasattr(self.connector, "query_with_confidence"):
                text, confidence = self.connector.query_with_confidence(
                    system_prompt, user_prompt
                )
            else:
                text = self.connector.query(system_prompt, user_prompt)
                confidence = 0.5
        except Exception as e:
            return (f"[대도서관 오류: {e}]", 0.0)

        if self.auto_distill and text and confidence > 0.1:
            self.buffer.add(DistillSample(
                input_text=user_prompt,
                target_text=text,
                source="library",
                confidence=confidence,
            ))

        return (text, confidence)

    def get_training_corpus(self, min_confidence: float = 0.3) -> str:
        """DynLLM 학습용 코퍼스 텍스트 반환."""
        texts = self.buffer.get_training_texts(min_confidence)
        return " ".join(texts)

    def status(self) -> dict:
        return {
            "connector": type(self.connector).__name__ if self.connector else "None",
            "query_count": self._query_count,
            "buffer": self.buffer.stats(),
            "auto_distill": self.auto_distill,
        }
