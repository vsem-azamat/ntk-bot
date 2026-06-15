"""Anchor guardrail: shift a forecast band so its median passes through the latest
observed sample, with the correction decaying over the remaining horizon. Keeps
the live plot connected to reality without re-shaping the model's trend."""


def anchor_band(
    p10: list[float],
    p50: list[float],
    p90: list[float],
    anchor_index: int | None,
    anchor_value: float | None,
    decay: float = 0.85,
) -> tuple[list[float], list[float], list[float]]:
    """Add a geometrically decaying correction so ``p50[anchor_index]`` equals
    ``anchor_value``. The same correction is applied to p10/p90 so the band shifts
    rigidly, then each triple is re-sorted to stay monotone."""
    if anchor_index is None or anchor_value is None:
        return p10, p50, p90

    correction = anchor_value - p50[anchor_index]
    out10, out50, out90 = [], [], []
    for i in range(len(p50)):
        adj = 0.0 if i < anchor_index else correction * decay ** (i - anchor_index)
        triple = sorted((max(0.0, p10[i] + adj), max(0.0, p50[i] + adj), max(0.0, p90[i] + adj)))
        out10.append(triple[0])
        out50.append(triple[1])
        out90.append(triple[2])
    return out10, out50, out90
