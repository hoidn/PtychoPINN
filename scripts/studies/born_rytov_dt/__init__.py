"""Study-local helpers for the Born/Rytov diffraction tomography candidate lane.

This package is intentionally task-local: it owns operator-validation
helpers and the ``operator_validation.json`` artifact contract. Dataset
generation, training, and four-row preflight machinery live in separate
backlog items and must not be added here.
"""
