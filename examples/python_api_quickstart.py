"""Build and inspect the controlled Causal4D protocol through API v1."""

from __future__ import annotations

from causal4d.api.v1 import CounterfactualBenchmarkConfig, build_protocol


def main() -> None:
    config = CounterfactualBenchmarkConfig(
        frame_count=18,
        training_repeats=1,
        parameter_grid_count=3,
    )
    protocol = build_protocol(config)
    for object_protocol in protocol:
        print(
            object_protocol.graph_object.name,
            len(object_protocol.train_actions),
            object_protocol.validation_action.action_id,
            object_protocol.test_action.action_id,
        )


if __name__ == "__main__":
    main()
