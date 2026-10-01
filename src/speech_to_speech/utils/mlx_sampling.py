"""Seed MLX sampling on the thread executing a model generation.

Call only while holding the shared MLX lock: the model module's sampler is
temporarily replaced and restored when generation finishes or is cancelled.
"""

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def seeded_mlx_sampling(generation_fn: Callable[..., Any], seed: int | None) -> Iterator[None]:
    if seed is None:
        yield
        return

    import mlx.core as mx

    mx.random.seed(seed)
    module = sys.modules.get(generation_fn.__module__)
    original_sampler = getattr(module, "categorical_sampling", None)
    if module is None or original_sampler is None:
        yield
        return

    # mlx-lm compiles its sampler at import time with that thread's RNG state.
    # MLX 0.32 has thread-local RNG state, so importing on the setup thread then
    # generating on a worker otherwise ignores the worker's seeded state.
    # Compile the same operation with the current worker's state instead.
    worker_sampler = mx.compile(
        lambda logits, temp: mx.random.categorical(logits * (1 / temp)),
        inputs=mx.random.state,
        outputs=mx.random.state,
    )
    setattr(module, "categorical_sampling", worker_sampler)
    try:
        yield
    finally:
        setattr(module, "categorical_sampling", original_sampler)
