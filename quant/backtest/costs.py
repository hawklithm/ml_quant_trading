from dataclasses import dataclass


@dataclass(frozen=True)
class TransactionCostModel:
    """Simple explicit cost model for long-only daily backtests."""

    commission_bps: float = 1.0
    slippage_bps: float = 5.0
    sell_tax_bps: float = 0.0

    def validate(self) -> None:
        values = (self.commission_bps, self.slippage_bps, self.sell_tax_bps)
        if any(value < 0 for value in values):
            raise ValueError("transaction costs must be non-negative")

    def rate(self, side: str) -> float:
        self.validate()
        normalized = side.upper()
        if normalized not in {"BUY", "SELL"}:
            raise ValueError(f"unsupported side: {side}")
        tax = self.sell_tax_bps if normalized == "SELL" else 0.0
        return (self.commission_bps + self.slippage_bps + tax) / 10_000.0

    def amount(self, notional: float, side: str) -> float:
        if notional < 0:
            raise ValueError("notional must be non-negative")
        return float(notional) * self.rate(side)
