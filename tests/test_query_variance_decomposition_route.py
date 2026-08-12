from causal4d.cli.command_registry import find_command


def test_query_variance_decomposition_route_is_diagnostic() -> None:
    command = find_command("diagnostic uncertainty decompose-query")

    assert command.target == "causal4d.cli.query_variance_decomposition:main"
    assert command.lifecycle == "diagnostic"
    assert command.claim_bearing is False
    assert command.historical_name is None
