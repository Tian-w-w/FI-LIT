from __future__ import annotations

from fi_lit.train import _cleanup_distributed_process_group, _tokenize_completion_only


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


class _FakeChatTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is True
        if add_generation_prompt:
            return [10, 11, 12]
        assert messages[-1]["role"] == "assistant"
        return [10, 11, 12, 20, 21, 99]


def test_completion_only_tokenization_masks_user_tokens() -> None:
    encoded = _tokenize_completion_only(
        {"definition": ["Classify."], "input": "text", "references": ["label"]},
        _FakeChatTokenizer(),
        max_seq_length=16,
    )
    assert encoded["input_ids"] == [10, 11, 12, 20, 21, 99]
    assert encoded["labels"] == [-100, -100, -100, 20, 21, 99]
