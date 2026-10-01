import sys
from types import ModuleType, SimpleNamespace

import pytest

from speech_to_speech.utils.mlx_sampling import seeded_mlx_sampling


@pytest.mark.parametrize("finish", ["normal", "error", "cancel"])
def test_sampler_uses_current_thread_rng_and_restores_on_exit(monkeypatch, finish):
    calls = []
    state = ["worker RNG"]
    mx = ModuleType("mlx.core")
    mx.random = SimpleNamespace(
        state=state,
        seed=lambda seed: calls.append(("seed", seed)),
        categorical=lambda logits: ("sample", logits),
    )
    compiled = []

    def compile_sampler(fn, *, inputs, outputs):
        assert inputs is state and outputs is state
        compiled.append(fn)
        return fn

    mx.compile = compile_sampler
    mlx = ModuleType("mlx")
    mlx.core = mx
    module = ModuleType("test_worker_sampling_module")
    original = object()
    module.categorical_sampling = original
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def generation():
        with seeded_mlx_sampling(generation, 42):
            assert module.categorical_sampling(4.0, 2.0) == ("sample", 2.0)
            yield "audio"
            if finish == "error":
                raise RuntimeError("generation failed")

    generation.__module__ = module.__name__
    stream = generation()
    assert next(stream) == "audio"
    if finish == "cancel":
        stream.close()
    elif finish == "error":
        with pytest.raises(RuntimeError, match="generation failed"):
            next(stream)
    else:
        assert list(stream) == []
    assert module.categorical_sampling is original
    assert len(compiled) == 1
    assert calls == [("seed", 42)]


def test_unseeded_generation_keeps_existing_sampler():
    # No MLX import or sampler override is needed for ordinary voice profiles.
    with seeded_mlx_sampling(lambda: None, None):
        pass
