"""
Shim for short import path: the v2 Modal app is implemented in modal_mls_app_v2.py.

Run:
  modal run modal_v2.py::main -- --m 0 --p 1
  modal run modal_v2.py::run_matrix_batch
"""

from modal_mls_app_v2 import app, main, run_matrix_batch, run_research_cell

__all__ = ["app", "main", "run_matrix_batch", "run_research_cell"]
