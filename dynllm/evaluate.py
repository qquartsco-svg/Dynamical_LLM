"""
DynLLM 평가 도구

언어 모델 지표:
  - Perplexity (next-token prediction 품질)
  - Generation diversity (생성 다양성)
  - Memory utilization (메모리 활용도)
  - Memory recall rate (기억 회상 적합도)
  - Adaptation gain (온라인 적응 효과)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .model import DynLLM
from .tokenizer import DynTokenizer


def compute_perplexity(
    model: DynLLM,
    text: str,
    tokenizer: DynTokenizer,
    seq_len: int = 64,
) -> float:
    """
    텍스트에 대한 perplexity 계산.
    낮을수록 예측 품질이 좋다.
    """
    model.eval()
    device = next(model.parameters()).device

    ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    if len(ids) < 2:
        return float("inf")

    total_loss = 0.0
    n_tokens = 0

    for start in range(0, len(ids) - seq_len, seq_len):
        chunk = ids[start: start + seq_len + 1]
        if len(chunk) < 2:
            break

        x = torch.tensor(chunk[:-1], dtype=torch.long).unsqueeze(0).to(device)
        y = torch.tensor(chunk[1:], dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, model.cfg.vocab_size),
                y.reshape(-1),
                reduction="sum",
            )

        total_loss += loss.item()
        n_tokens += y.numel()

    if n_tokens == 0:
        return float("inf")

    avg_loss = total_loss / n_tokens
    return math.exp(min(avg_loss, 20))


def compute_generation_diversity(
    model: DynLLM,
    tokenizer: DynTokenizer,
    prompts: list[str],
    max_tokens: int = 50,
    n_samples: int = 3,
) -> dict:
    """
    생성 다양성 측정.
    같은 프롬프트에서 여러 번 생성했을 때 얼마나 다양한지.
    """
    model.eval()

    all_unique = 0
    all_total = 0

    for prompt in prompts:
        generations = set()
        prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        for _ in range(n_samples):
            gen_ids = model.generate(prompt_ids, max_tokens=max_tokens, temperature=0.9)
            gen_text = tokenizer.decode(gen_ids)
            generations.add(gen_text)
        all_unique += len(generations)
        all_total += n_samples

    return {
        "unique_ratio": all_unique / max(all_total, 1),
        "total_samples": all_total,
        "unique_samples": all_unique,
    }


def compute_memory_utilization(model: DynLLM) -> dict:
    """메모리 시스템 활용도 진단."""
    if model.memory is None:
        return {"has_memory": False}

    heb = model.memory.hebbian
    ep = model.memory.episodic

    return {
        "has_memory": True,
        "hebbian_stored": heb.n_stored.item(),
        "hebbian_capacity": heb.capacity,
        "hebbian_utilization": heb.n_stored.item() / max(heb.capacity, 1),
        "hebbian_W_norm": heb.W.norm().item(),
        "episodic_stored": ep.n_stored.item(),
        "episodic_capacity": ep.n_episodes,
        "episodic_utilization": min(ep.n_stored.item(), ep.n_episodes) / max(ep.n_episodes, 1),
    }


def compute_memory_recall_rate(
    model: DynLLM,
    cues: list[torch.Tensor],
    targets: list[torch.Tensor],
) -> dict:
    """
    메모리 회상 적합도 평가.

    cue -> recalled pattern 이 target과 얼마나 닮았는지 cosine similarity로 측정한다.
    Hebbian selective recall과 episodic recall을 분리해서 본다.
    """
    if model.memory is None:
        return {
            "has_memory": False,
            "n_trials": 0,
            "hebbian_recall_rate_0_1": 0.0,
            "episodic_recall_rate_0_1": 0.0,
            "overall_recall_rate_0_1": 0.0,
        }

    if len(cues) != len(targets):
        raise ValueError("cues and targets must have the same length")

    if not cues:
        return {
            "has_memory": True,
            "n_trials": 0,
            "hebbian_recall_rate_0_1": 0.0,
            "episodic_recall_rate_0_1": 0.0,
            "overall_recall_rate_0_1": 0.0,
        }

    device = next(model.parameters()).device
    hebbian_scores: list[float] = []
    episodic_scores: list[float] = []

    for cue, target in zip(cues, targets):
        cue_b = cue.to(device).reshape(1, -1)
        target_b = target.to(device).reshape(1, -1)

        with torch.no_grad():
            hebbian = model.memory.hebbian.selective_recall(cue_b)
            episodic = model.memory.episodic.recall_episode(cue_b)

        heb_score = F.cosine_similarity(hebbian, target_b, dim=-1).mean().item()
        epi_score = F.cosine_similarity(episodic, target_b, dim=-1).mean().item()

        hebbian_scores.append((heb_score + 1.0) / 2.0)
        episodic_scores.append((epi_score + 1.0) / 2.0)

    heb_mean = sum(hebbian_scores) / len(hebbian_scores)
    epi_mean = sum(episodic_scores) / len(episodic_scores)
    overall = 0.5 * (heb_mean + epi_mean)

    return {
        "has_memory": True,
        "n_trials": len(cues),
        "hebbian_recall_rate_0_1": heb_mean,
        "episodic_recall_rate_0_1": epi_mean,
        "overall_recall_rate_0_1": overall,
    }


def full_evaluation(
    model: DynLLM,
    tokenizer: DynTokenizer,
    eval_text: str,
    prompts: Optional[list[str]] = None,
) -> dict:
    """전체 평가 리포트."""
    ppl = compute_perplexity(model, eval_text, tokenizer)

    if prompts is None:
        prompts = [eval_text[:10], eval_text[10:20], eval_text[20:30]]

    diversity = compute_generation_diversity(model, tokenizer, prompts)
    memory = compute_memory_utilization(model)

    return {
        "perplexity": ppl,
        "diversity": diversity,
        "memory": memory,
        "parameters": model.count_parameters(),
    }
