"""Scan pipeline: capture -> reconstruct -> detect+localize -> map -> persist.

Every stage is a pure, swappable unit with no GUI and no global state. On failure a
stage raises ``PipelineError(stage, reason)`` — it never substitutes fake output.
Mock implementations (for DEMO_MODE / tests) are clearly labeled as synthetic.
"""
