# BarSchema (FEATURES.md 2.2): the caller-supplied map from column roles to physical column names.
# Features declare roles; `.column(role)` resolves them at build time. sabia adjusts and infers
# nothing -- the schema records the adjustment basis each column carries, and the manifest pins it.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from sabia.typing import (
    Adjustment,
    FlowField,
    FlowRole,
    InputRole,
    PriceField,
    PriceRole,
    QuoteField,
    QuoteRole,
    VolumeField,
    VolumeRole,
)

# The canonical bars-closed marker column (FEATURES.md 8.3). The tick->bar adapter emits it and
# ``trades()``/``quotes()`` map it by default, so validate()'s finalization gate is armed out of
# the box on adapter output; on frames that simply lack the column the gate is skipped.
DEFAULT_CLOSED_COLUMN = "closed"


@dataclass(frozen=True, slots=True)
class BarSchema:
    """Resolves column roles to physical columns for one frame/manifest (FEATURES.md 2.2, 2.4).

    ``symbol`` and ``timestamp`` are fixed canonical column names (FEATURES.md 2.1), not
    role-tagged; the OHLCV/factor columns are role-tagged. ``calendar`` is the exchange code (one
    per frame, 2.4); ``closed_col`` is the bars-closed marker (8.3). Frozen, with ``roles`` wrapped
    in an immutable view so the guarantee holds even if the caller mutates the dict they passed.
    """

    roles: Mapping[InputRole, str]
    closed_col: str | None = None
    calendar: str = "UTC"
    symbol_col: str = "symbol"
    timestamp_col: str = "timestamp"

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", MappingProxyType(dict(self.roles)))

    @classmethod
    def ohlcv(
        cls,
        *,
        open: str = "open",
        high: str = "high",
        low: str = "low",
        close: str = "close",
        volume: str = "volume",
        tr_close: str | None = None,
        symbol_col: str = "symbol",
        timestamp_col: str = "timestamp",
        closed_col: str | None = None,
        calendar: str = "UTC",
    ) -> BarSchema:
        """Build a schema from plain OHLCV column names -- the common case, no hand-rolled roles.

        Maps the OHLC columns to the **split-only** basis (the range-safe basis the estimators use,
        FEATURES.md 2.2) and ``volume`` to volume@split.

        IMPORTANT -- the @tr conflation when ``tr_close`` is omitted: ``open@tr`` and ``close@tr``
        are then backed by the **same split-only columns** as ``open@split`` / ``close@split``.
        There is no separate total-return series, so returns / momentum / trend features (which
        request the ``@tr`` roles) will silently run on split-only prices that are *labelled* total
        return. On dividend-paying instruments this is wrong -- the dividend drop is treated as a
        real return. Pass ``tr_close`` (e.g. an ``adj_close`` column) whenever a distinct
        total-return close exists, so the ``@tr`` roles resolve to the adjusted series.

        For richer inputs (VWAP, dollar volume, a market factor) construct
        ``BarSchema(roles={...})`` directly -- this covers OHLCV only.
        """
        close_tr_col = tr_close if tr_close is not None else close
        roles: dict[InputRole, str] = {
            PriceRole(PriceField.OPEN, Adjustment.SPLIT): open,
            PriceRole(PriceField.HIGH, Adjustment.SPLIT): high,
            PriceRole(PriceField.LOW, Adjustment.SPLIT): low,
            PriceRole(PriceField.CLOSE, Adjustment.SPLIT): close,
            VolumeRole(VolumeField.VOLUME, Adjustment.SPLIT): volume,
            PriceRole(PriceField.OPEN, Adjustment.TR): open,
            PriceRole(PriceField.CLOSE, Adjustment.TR): close_tr_col,
        }
        return cls(
            roles=roles,
            closed_col=closed_col,
            calendar=calendar,
            symbol_col=symbol_col,
            timestamp_col=timestamp_col,
        )

    @staticmethod
    def _raw_trade_roles(
        *,
        open: str,
        high: str,
        low: str,
        close: str,
        volume: str,
        vwap: str | None,
        trade_count: str | None,
        signed_volume: str | None,
        buy_volume: str | None,
        sell_volume: str | None,
        dollar_volume: str | None,
        signed_dollar: str | None,
    ) -> dict[InputRole, str]:
        # The shared raw-basis OHLCV + optional flow role map for trades()/quotes(). Intraday micro
        # works on the RAW (traded) basis -- one session has no split/dividend boundary. Optional
        # columns are mapped only when supplied (``is not None``); an unmapped role that a feature
        # needs then fails loudly at resolution (FEATURES.md 4.2), never silently.
        roles: dict[InputRole, str] = {
            PriceRole(PriceField.OPEN, Adjustment.RAW): open,
            PriceRole(PriceField.HIGH, Adjustment.RAW): high,
            PriceRole(PriceField.LOW, Adjustment.RAW): low,
            PriceRole(PriceField.CLOSE, Adjustment.RAW): close,
            VolumeRole(VolumeField.VOLUME, Adjustment.RAW): volume,
        }
        optional: dict[InputRole, str | None] = {
            PriceRole(PriceField.VWAP, Adjustment.RAW): vwap,
            VolumeRole(VolumeField.DOLLAR_VOLUME, Adjustment.RAW): dollar_volume,
            FlowRole(FlowField.TRADE_COUNT, Adjustment.RAW): trade_count,
            FlowRole(FlowField.SIGNED_VOLUME, Adjustment.RAW): signed_volume,
            FlowRole(FlowField.BUY_VOLUME, Adjustment.RAW): buy_volume,
            FlowRole(FlowField.SELL_VOLUME, Adjustment.RAW): sell_volume,
            FlowRole(FlowField.SIGNED_DOLLAR, Adjustment.RAW): signed_dollar,
        }
        roles.update({role: col for role, col in optional.items() if col is not None})
        return roles

    @classmethod
    def trades(
        cls,
        *,
        open: str = "open",
        high: str = "high",
        low: str = "low",
        close: str = "close",
        volume: str = "volume",
        vwap: str | None = None,
        trade_count: str | None = None,
        signed_volume: str | None = None,
        buy_volume: str | None = None,
        sell_volume: str | None = None,
        dollar_volume: str | None = None,
        signed_dollar: str | None = None,
        symbol_col: str = "symbol",
        timestamp_col: str = "timestamp",
        closed_col: str | None = DEFAULT_CLOSED_COLUMN,
        calendar: str = "UTC",
    ) -> BarSchema:
        """Build a schema for intraday trade bars -- OHLCV on the raw basis + optional flow columns.

        The common output shape of the ``sabia.adapters`` tick->bar layer (FEATURES.md 13): OHLCV
        plus the adapter-derived flow aggregates (``signed_volume``, ``buy_volume`` ...). Map only
        the optional columns your bars actually carry; a microstructure feature that needs an
        unmapped role fails loudly at resolution rather than reading the wrong column.

        ``closed_col`` defaults to the adapter's ``closed`` marker, arming validate()'s
        finalization gate on adapter output (the in-progress trailing bar is rejected until
        filtered); frames without the column skip the gate as before.

        These are **raw** (traded) prices -- correct within one session, but they must not be mixed
        into a daily cross-asset panel that expects the @split / @tr bases (cf. ``ohlcv``).
        """
        return cls(
            roles=cls._raw_trade_roles(
                open=open,
                high=high,
                low=low,
                close=close,
                volume=volume,
                vwap=vwap,
                trade_count=trade_count,
                signed_volume=signed_volume,
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                dollar_volume=dollar_volume,
                signed_dollar=signed_dollar,
            ),
            closed_col=closed_col,
            calendar=calendar,
            symbol_col=symbol_col,
            timestamp_col=timestamp_col,
        )

    @classmethod
    def quotes(
        cls,
        *,
        bid: str = "bid",
        ask: str = "ask",
        bid_size: str | None = None,
        ask_size: str | None = None,
        mid: str | None = None,
        open: str = "open",
        high: str = "high",
        low: str = "low",
        close: str = "close",
        volume: str = "volume",
        vwap: str | None = None,
        trade_count: str | None = None,
        signed_volume: str | None = None,
        buy_volume: str | None = None,
        sell_volume: str | None = None,
        dollar_volume: str | None = None,
        signed_dollar: str | None = None,
        symbol_col: str = "symbol",
        timestamp_col: str = "timestamp",
        closed_col: str | None = DEFAULT_CLOSED_COLUMN,
        calendar: str = "UTC",
    ) -> BarSchema:
        """Build a schema for intraday trade bars enriched with L1 quotes (FEATURES.md 13, tier L1).

        ``trades`` plus the best bid/ask (and optional sizes / mid). ``bid``/``ask`` map by default;
        the sizes and mid are mapped only when supplied. Quote prices are raw, like trade prices.
        ``closed_col`` defaults to the adapter's ``closed`` marker (cf. ``trades``).
        """
        roles = cls._raw_trade_roles(
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
            vwap=vwap,
            trade_count=trade_count,
            signed_volume=signed_volume,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            dollar_volume=dollar_volume,
            signed_dollar=signed_dollar,
        )
        roles[QuoteRole(QuoteField.BID, Adjustment.RAW)] = bid
        roles[QuoteRole(QuoteField.ASK, Adjustment.RAW)] = ask
        optional_quotes: dict[InputRole, str | None] = {
            QuoteRole(QuoteField.BID_SIZE, Adjustment.RAW): bid_size,
            QuoteRole(QuoteField.ASK_SIZE, Adjustment.RAW): ask_size,
            QuoteRole(QuoteField.MID, Adjustment.RAW): mid,
        }
        roles.update({role: col for role, col in optional_quotes.items() if col is not None})
        return cls(
            roles=roles,
            closed_col=closed_col,
            calendar=calendar,
            symbol_col=symbol_col,
            timestamp_col=timestamp_col,
        )

    def column(self, role: InputRole) -> str:
        """Physical column backing ``role``; raises a precise ``KeyError`` if undeclared."""
        try:
            return self.roles[role]
        except KeyError:
            declared = sorted(str(r) for r in self.roles)
            raise KeyError(
                f"BarSchema has no column for role {role}; declared roles: {declared}"
            ) from None

    def has(self, role: InputRole) -> bool:
        return role in self.roles


__all__ = ["DEFAULT_CLOSED_COLUMN", "BarSchema"]
