"""Analytic FLOP accounting for autoregressive generation.

The standard dense-transformer forward-pass approximation is ``2 * N`` floating
point operations per token, where ``N`` is the number of model parameters (one
multiply and one add per parameter).  For a prompt of ``P`` tokens followed by a
completion of ``C`` tokens we report::

    FLOPs = 2 * N * (P + C)

This counts the prefill forward pass and the forward pass for every generated
token.  It deliberately excludes the quadratic attention term, the verifier's
own compute, and the backwards pass during verifier training.  Those choices are
documented in the paper so that all strategy comparisons use the same accounting
convention.
"""

from __future__ import annotations

from numbers import Real


def estimate_flops(n_params: int, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate inference FLOPs with the ``2*N`` per-token approximation.

    Parameters
    ----------
    n_params:
        Number of model parameters.
    prompt_tokens:
        Number of prompt/prefill tokens processed.
    completion_tokens:
        Number of generated completion tokens.

    Returns
    -------
    float
        Estimated forward-pass FLOPs.
    """
    if isinstance(n_params, bool) or not isinstance(n_params, Real) or n_params <= 0:
        raise ValueError("n_params must be a positive number")
    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError("token counts must be non-negative")
    return float(2.0 * n_params * (prompt_tokens + completion_tokens))


def cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
) -> float:
    """Convert token counts to dollars using per-million-token prices."""
    if min(prompt_tokens, completion_tokens, input_price_per_million, output_price_per_million) < 0:
        raise ValueError("token counts and prices must be non-negative")
    return (
        prompt_tokens * input_price_per_million
        + completion_tokens * output_price_per_million
    ) / 1_000_000.0
