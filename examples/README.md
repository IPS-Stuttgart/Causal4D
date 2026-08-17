# Examples

## Controlled protocol construction

```bash
python examples/python_api_quickstart.py
```

This prints the objects and held-out actions in the small controlled benchmark.

## End-to-end abduction, intervention, and prediction

```bash
python -m causal4d.demo.aip --output-dir build/aip-demo
```

The module runs with the core NumPy/SciPy installation and writes verified,
non-pickled Causal4D contracts for a `TwinBelief`, factual intervention,
counterfactual query, dense physical posterior, and projected posterior. It also
checks that changing held-out suffix frames cannot change factual abduction.

The equivalent repository wrapper is:

```bash
python examples/aip_end_to_end.py --output-dir build/aip-demo
```

The output is a controlled software demonstration. It is not physical evidence,
a provider-competence result, or a scientific claim.
