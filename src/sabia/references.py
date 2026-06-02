# Structured citations (FEATURES.md 4.1.1): a feature separates its *formula* provenance (who
# defined the computation) from its *empirical* anchors (the anomaly evidence). Pure-measurement
# features carry only a formula reference; replicated anomalies add empirical ones. Citations are
# metadata -- they do NOT fold into the fingerprint (editing a citation must not force a bump).

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Reference:
    """One literature anchor, rendered ``Authors (Year)``."""

    authors: str
    year: int
    title: str | None = None

    def __str__(self) -> str:
        return f"{self.authors} ({self.year})"


@dataclass(frozen=True, slots=True)
class Citation:
    """Formula provenance + optional empirical anchors (FEATURES.md 4.1.1)."""

    formula: Reference
    empirical: tuple[Reference, ...] = ()

    def __str__(self) -> str:
        tail = "; ".join(str(ref) for ref in self.empirical)
        return f"{self.formula}" + (f" — {tail}" if tail else "")


__all__ = ["Citation", "Reference"]
