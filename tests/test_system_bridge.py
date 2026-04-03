from dynllm.system_bridge import (
    AthenaRecommendation,
    AthenaStage,
    DecisionRoute,
    DynLLMDraft,
    GovernanceMode,
    build_governance_packet,
)


def test_standalone_positive_routes_to_user():
    packet = build_governance_packet(
        GovernanceMode.STANDALONE_PERSONAL,
        DynLLMDraft(text="draft"),
        AthenaRecommendation(
            stage=AthenaStage.POSITIVE,
            advisory="ok",
            allow_execute=True,
        ),
    )
    assert packet.route == DecisionRoute.DIRECT_TO_USER


def test_standalone_cautious_routes_via_athena():
    packet = build_governance_packet(
        GovernanceMode.STANDALONE_PERSONAL,
        DynLLMDraft(text="draft"),
        AthenaRecommendation(
            stage=AthenaStage.CAUTIOUS,
            advisory="be careful",
            allow_execute=False,
        ),
    )
    assert packet.route == DecisionRoute.ATHENA_TO_USER


def test_nexus_cautious_routes_to_pharaoh():
    packet = build_governance_packet(
        GovernanceMode.NEXUS_CONNECTED,
        DynLLMDraft(text="draft"),
        AthenaRecommendation(
            stage=AthenaStage.CAUTIOUS,
            advisory="needs consensus",
            allow_execute=False,
        ),
    )
    assert packet.route == DecisionRoute.ATON_TO_PHARAOH_TO_USER


def test_negative_holds_and_revises():
    packet = build_governance_packet(
        GovernanceMode.NEXUS_CONNECTED,
        DynLLMDraft(text="draft"),
        AthenaRecommendation(
            stage=AthenaStage.NEGATIVE,
            advisory="reject",
            allow_execute=False,
        ),
    )
    assert packet.route == DecisionRoute.HOLD_AND_REVISE
