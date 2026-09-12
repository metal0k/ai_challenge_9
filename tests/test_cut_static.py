"""Pure-function coverage for tools/cut_static.py — no ffmpeg, no real video.

Everything that touches a subprocess (probe_duration, run_freezedetect,
run_cut, has_audio_stream) is deliberately left untested here: the module is
structured so the freeze-list math and the filter-string building are plain
functions, and that is what this file exercises.
"""

from __future__ import annotations

from tools import cut_static


def test_parses_a_multi_freeze_block_with_a_trailing_open_freeze():
    """freeze_duration lines are noise once start/end are paired; the last
    freeze_start never closes and must clamp to the probed duration."""
    stderr_text = "\n".join(
        [
            "[freezedetect @ 0x1] lavfi.freezedetect.freeze_start: 12.345",
            "[freezedetect @ 0x1] lavfi.freezedetect.freeze_duration: 30.200",
            "[freezedetect @ 0x1] lavfi.freezedetect.freeze_end: 42.545",
            "[freezedetect @ 0x1] lavfi.freezedetect.freeze_start: 50.000",
            "[freezedetect @ 0x1] lavfi.freezedetect.freeze_duration: 10.000",
            "[freezedetect @ 0x1] lavfi.freezedetect.freeze_end: 60.000",
            "[freezedetect @ 0x1] lavfi.freezedetect.freeze_start: 90.000",
        ]
    )

    freezes = cut_static.parse_freezedetect(stderr_text, duration=100.0)

    assert freezes == [(12.345, 42.545), (50.0, 60.0), (90.0, 100.0)]


def test_no_freeze_lines_gives_an_empty_list():
    text = "[libx264 @ 0x1] frame I:1 Avg QP:20.0 size: 12345"

    assert cut_static.parse_freezedetect(text, duration=100.0) == []


def test_keep_interval_for_a_freeze_in_the_middle():
    intervals = cut_static.compute_keep_intervals([(10.0, 20.0)], duration=100.0, keep=1.5)

    assert intervals == [(0.0, 10.0), (10.0, 11.5), (20.0, 100.0)]


def test_keep_interval_for_a_freeze_starting_at_zero():
    intervals = cut_static.compute_keep_intervals([(0.0, 5.0)], duration=50.0, keep=1.5)

    assert intervals == [(0.0, 1.5), (5.0, 50.0)]


def test_keep_interval_for_two_adjacent_freezes():
    intervals = cut_static.compute_keep_intervals(
        [(10.0, 20.0), (20.0, 30.0)], duration=50.0, keep=1.5
    )

    assert intervals == [(0.0, 10.0), (10.0, 11.5), (20.0, 21.5), (30.0, 50.0)]


def test_a_freeze_shorter_than_keep_is_left_whole():
    intervals = cut_static.compute_keep_intervals([(10.0, 10.8)], duration=50.0, keep=1.5)

    assert intervals == [(0.0, 10.0), (10.0, 10.8), (10.8, 50.0)]


def test_a_freeze_running_to_the_end_of_file_drops_its_tail_only_when_asked():
    """The closing freeze is held by default (it is the final screen); the
    cut-everything behaviour is still available and pinned here."""
    held = cut_static.compute_keep_intervals([(80.0, 100.0)], duration=100.0, keep=1.5)
    cut = cut_static.compute_keep_intervals(
        [(80.0, 100.0)], duration=100.0, keep=1.5, hold_tail=False
    )

    assert held == [(0.0, 80.0), (80.0, 100.0)]
    assert cut == [(0.0, 80.0), (80.0, 81.5)]


def test_intervals_are_ordered_non_overlapping_and_never_past_duration():
    intervals = cut_static.compute_keep_intervals(
        [(0.0, 3.0), (10.0, 20.0), (20.0, 90.0)], duration=90.0, keep=1.5
    )

    # The last freeze ends at the probed duration, so it is the closing screen
    # and stays whole; the two before it are cut to `keep`.
    assert intervals == [(0.0, 1.5), (3.0, 10.0), (10.0, 11.5), (20.0, 90.0)]
    starts = [start for start, _ in intervals]
    assert starts == sorted(starts)
    for (_, prev_end), (next_start, _) in zip(intervals, intervals[1:], strict=False):
        assert prev_end <= next_start
    for _, end in intervals:
        assert end <= 90.0


def test_filter_complex_trims_both_streams_from_one_interval_list():
    """The bug this shape exists for: video cut to 97 s while audio stayed at
    the take's original 570 s, so the container still claimed the old length."""
    intervals = [(0.0, 10.0), (10.0, 11.5)]

    text = cut_static.build_filter_complex(intervals, with_audio=True)

    assert text.count("[0:v]trim=") == 2
    assert text.count("[0:a]atrim=") == 2
    assert "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]" in text
    assert "start=0:end=10" in text
    assert "start=10:end=11.5" in text


def test_filter_complex_without_audio_asks_concat_for_video_only():
    text = cut_static.build_filter_complex([(0.0, 10.0)], with_audio=False)

    assert "atrim" not in text
    assert "[v0]concat=n=1:v=1:a=0[v]" in text


def test_nothing_frozen_is_reported_as_nothing_to_cut():
    intervals = cut_static.compute_keep_intervals([], duration=100.0, keep=1.5)

    assert intervals == [(0.0, 100.0)]
    assert cut_static.cut_summary([], intervals, duration=100.0) == (
        "нечего вырезать: заморозок длиннее --keep не найдено"
    )


def test_a_freeze_too_short_to_trim_also_reports_nothing_to_cut():
    """Detected but shorter than --keep: compute_keep_intervals keeps it whole,
    and the summary must say so rather than claim a cut that never happened."""
    freezes = [(10.0, 10.8)]
    intervals = cut_static.compute_keep_intervals(freezes, duration=50.0, keep=1.5)

    assert cut_static.cut_summary(freezes, intervals, duration=50.0) == (
        "нечего вырезать: заморозок длиннее --keep не найдено"
    )


def test_an_actual_cut_is_reported_with_the_dropped_seconds():
    freezes = [(10.0, 20.0)]
    intervals = cut_static.compute_keep_intervals(freezes, duration=100.0, keep=1.5)

    summary = cut_static.cut_summary(freezes, intervals, duration=100.0)

    assert summary == "заморозок: 1, вырезано 8.5 с из 100.0 с"


def test_the_freeze_that_closes_the_take_is_kept_whole():
    """The last frozen stretch is the final screen — the numbers the viewer is
    meant to read. Trimming it to `keep` ends the video mid-thought."""
    intervals = cut_static.compute_keep_intervals([(10.0, 60.0)], duration=60.0, keep=1.5)

    assert intervals == [(0.0, 10.0), (10.0, 60.0)]


def test_a_closing_freeze_is_cut_like_any_other_when_hold_tail_is_off():
    intervals = cut_static.compute_keep_intervals(
        [(10.0, 60.0)], duration=60.0, keep=1.5, hold_tail=False
    )

    assert intervals == [(0.0, 10.0), (10.0, 11.5)]


def test_holding_the_tail_does_not_spare_a_freeze_in_the_middle():
    intervals = cut_static.compute_keep_intervals(
        [(10.0, 40.0), (50.0, 60.0)], duration=60.0, keep=1.5
    )

    assert intervals == [(0.0, 10.0), (10.0, 11.5), (40.0, 50.0), (50.0, 60.0)]
