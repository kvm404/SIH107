"""Thread-context mechanics: truncation limits and two-sided history."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bis_assistant import slots, threads


def test_slots_public_accessor():
    assert slots.active_slots() is slots._active()


def test_threads_normalize_and_reset():
    assert threads.normalize_context(None) == {"history": [], "rounds": 0, "force": False}
    assert threads.normalize_context({"history": ["a"], "rounds": "2", "force": 1}) == {
        "history": ["a"], "rounds": 2, "force": True}
    assert threads.reset_context() == {"history": [], "rounds": 0, "force": False}
    assert threads.with_force({"history": ["a"], "rounds": 1, "force": False})["force"] is True


def test_threads_truncation_limits():
    assert threads.HISTORY_LIMIT == 8 and threads.BRIDGE_LIMIT == 4
    assert threads.push_history(["a", "b"], "c") == ["a", "b", "c"]
    assert threads.push_history([str(i) for i in range(10)], "x") == [str(i) for i in range(3, 10)] + ["x"]
    assert threads.bridge_history([str(i) for i in range(6)]) == ["2", "3", "4", "5"]
    assert threads.rounds_from(None, default=7) == 7
    assert threads.rounds_from({"rounds": 3}, default=7) == 3
    assert threads.combined_query(["steel bottle"], "vacuum") == "steel bottle vacuum"


def test_push_turn_keeps_both_sides_for_follow_ups():
    h = threads.push_turn([], "Is ISI mandatory for pressure cookers?", "Yes, under IS 2347.")
    assert h == ["User: Is ISI mandatory for pressure cookers?",
                 "Assistant: Yes, under IS 2347."]
    # Failed turns (no assistant text) keep only the user message.
    assert threads.push_turn(h, "which labs test it?", "")[-1] == "User: which labs test it?"
    long = threads.push_turn([f"x{i}" for i in range(8)], "q", "a")
    assert len(long) == threads.HISTORY_LIMIT and long[-2:] == ["User: q", "Assistant: a"]
