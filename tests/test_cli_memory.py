from pathlib import Path
from types import SimpleNamespace

import week_02.cli as cli
from advent_core.memory import MemorySnapshot, MemoryStore, StructuredMemory


def _shell(tmp_path: Path) -> cli.AgentShell:
    shell = cli.AgentShell.__new__(cli.AgentShell)
    shell.memory_store = MemoryStore(tmp_path)
    shell.session = SimpleNamespace(name="default")
    shell.memory = MemorySnapshot(
        working=StructuredMemory({"goal.primary": "ship"}),
        long_term=StructuredMemory({"preferences.language": "ru"}),
    )
    shell.memory_upto = 2
    shell.memory_dirty_working = True
    shell.memory_dirty_long_term = True
    return shell


def test_memory_layer_aliases_are_normalized() -> None:
    assert cli._memory_layer_name("short-term") == "short"
    assert cli._memory_layer_name("short_term") == "short"
    assert cli._memory_layer_name("long-term") == "long"
    assert cli._memory_layer_name("LONG_TERM") == "long"


def test_save_memory_retries_failed_layer_independently(tmp_path, monkeypatch) -> None:
    shell = _shell(tmp_path)

    def fail_working(*args, **kwargs):
        raise OSError("read-only")

    original = MemoryStore.save_working
    monkeypatch.setattr(MemoryStore, "save_working", fail_working)
    shell._save_memory()

    assert shell.memory_dirty_working is True
    assert shell.memory_dirty_long_term is False
    assert shell.memory_store.long_term_path.is_file()

    monkeypatch.setattr(MemoryStore, "save_working", original)
    shell._save_memory()
    assert shell.memory_dirty_working is False
    assert shell.memory_store.working_path("default").is_file()
