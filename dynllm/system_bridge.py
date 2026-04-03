"""
System bridge contracts for DynLLM <-> Atom / Athena / Aton / Pharaoh / User.

DynLLM is treated here as a cortical language-state core, not as the whole agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GovernanceMode(str, Enum):
    STANDALONE_PERSONAL = "standalone_personal"
    NEXUS_CONNECTED = "nexus_connected"


class AthenaStage(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    CAUTIOUS = "cautious"
    NEGATIVE = "negative"


class DecisionRoute(str, Enum):
    DIRECT_TO_USER = "direct_to_user"
    ATHENA_TO_USER = "athena_to_user"
    ATON_TO_PHARAOH_TO_USER = "aton_to_pharaoh_to_user"
    HOLD_AND_REVISE = "hold_and_revise"


@dataclass
class DynLLMDraft:
    """Raw cortical draft produced by DynLLM before governance review."""

    text: str
    confidence_0_1: float = 0.5
    state_summary: str = ""
    memory_sources: list[str] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)
    action_hints: list[str] = field(default_factory=list)


@dataclass
class AthenaRecommendation:
    """Supervisory recommendation produced by ATHENA."""

    stage: AthenaStage
    advisory: str
    allow_execute: bool
    requires_public_consensus: bool = False
    evidence_tags: list[str] = field(default_factory=list)


@dataclass
class PharaohOpinion:
    """Collected opinion or public perspective for consensus mode."""

    source: str
    stance: str
    weight_0_1: float = 0.5
    note: str = ""


@dataclass
class GovernancePacket:
    """
    Unified packet that can travel through Atom -> Athena -> Aton -> Pharaoh -> User.
    """

    mode: GovernanceMode
    draft: DynLLMDraft
    athena: AthenaRecommendation
    route: DecisionRoute
    pharaoh_opinions: list[PharaohOpinion] = field(default_factory=list)
    final_user_required: bool = True


def determine_route(
    mode: GovernanceMode,
    recommendation: AthenaRecommendation,
) -> DecisionRoute:
    """
    Decide how the system should route a draft.

    Standalone mode:
      Atom/Athena can act as independent intelligences, but the user still decides.

    Nexus mode:
      The packet is elevated into ATON orchestration and can move into Pharaoh consensus.
    """
    if recommendation.stage == AthenaStage.NEGATIVE:
        return DecisionRoute.HOLD_AND_REVISE

    if mode == GovernanceMode.STANDALONE_PERSONAL:
        if recommendation.stage == AthenaStage.CAUTIOUS:
            return DecisionRoute.ATHENA_TO_USER
        return DecisionRoute.DIRECT_TO_USER

    if recommendation.requires_public_consensus or recommendation.stage == AthenaStage.CAUTIOUS:
        return DecisionRoute.ATON_TO_PHARAOH_TO_USER

    return DecisionRoute.ATHENA_TO_USER


def build_governance_packet(
    mode: GovernanceMode,
    draft: DynLLMDraft,
    athena: AthenaRecommendation,
    pharaoh_opinions: list[PharaohOpinion] | None = None,
) -> GovernancePacket:
    route = determine_route(mode, athena)
    return GovernancePacket(
        mode=mode,
        draft=draft,
        athena=athena,
        route=route,
        pharaoh_opinions=pharaoh_opinions or [],
        final_user_required=True,
    )

