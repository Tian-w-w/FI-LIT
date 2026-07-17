from __future__ import annotations

from fi_lit.train import _cleanup_distributed_process_group


class _FakeDistributed:
    def __init__(self, initialized: bool) -> None:
        self.initialized = initialized
        self.destroyed = False

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return self.initialized

    def destroy_process_group(self) -> None:
        self.destroyed = True


class _FakeTorch:
    def __init__(self, initialized: bool) -> None:
        self.distributed = _FakeDistributed(initialized)


def test_cleanup_destroys_initialized_group() -> None:
    torch_module = _FakeTorch(initialized=True)
    _cleanup_distributed_process_group(torch_module)
    assert torch_module.distributed.destroyed is True


def test_cleanup_skips_uninitialized_group() -> None:
    torch_module = _FakeTorch(initialized=False)
    _cleanup_distributed_process_group(torch_module)
    assert torch_module.distributed.destroyed is False
