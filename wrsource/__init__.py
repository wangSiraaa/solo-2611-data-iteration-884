"""Revisionable weighted record source and reproducible weighted sampler.

Three independently runnable modules live in this package:

* :mod:`wrsource.source`  -- SQLite backed, versioned master source service.
* :mod:`wrsource.dataset` -- reproducible weighted sample / manifest producer.
* :mod:`wrsource.tunestore` / :mod:`wrsource.tuning` -- offline two-branch
  tuning-ticket merges, preview rehearsals and promise-bound issuance.

The implementation intentionally uses only the Python 3.11 standard library.
"""

__all__ = ["source", "sampling", "dataset", "tuning", "tunestore"]
__version__ = "1.1.0"
