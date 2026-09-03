from channel_compressor.utils import dedupe_adjacent_segments, parse_video_id


def test_parse_video_id_variants():
    expected = "PfTk7QiuJEc"
    assert parse_video_id(expected) == expected
    assert parse_video_id(f"https://www.youtube.com/watch?v={expected}&t=12s") == expected
    assert parse_video_id(f"https://youtu.be/{expected}") == expected
    assert parse_video_id(f"https://www.youtube.com/shorts/{expected}") == expected


def test_adjacent_caption_deduplication():
    segments = [
        {"text": "production beats passive consumption", "start": 0, "duration": 2},
        {"text": "production beats passive consumption", "start": 2, "duration": 2},
        {"text": "practice turns ideas into skill", "start": 4, "duration": 2},
    ]
    cleaned = dedupe_adjacent_segments(segments)
    assert [item["text"] for item in cleaned] == [
        "production beats passive consumption",
        "practice turns ideas into skill",
    ]
