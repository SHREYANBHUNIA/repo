import json
from pathlib import Path

from campus_bot import CampusInfoBot, CompressedCampusIndex


def test_build_and_load_index(tmp_path: Path):
    index_path = tmp_path / "campus_index.json.gz"
    index = CompressedCampusIndex(index_path=index_path)
    index.build(Path("data/course_catalog.json"), Path("data/campus_policies.json"))

    assert index_path.exists()
    stats = index.compression_stats()
    assert stats["compressed_bytes"] > 0
    assert stats["compressed_bytes"] < stats["raw_bytes"]

    loaded = CompressedCampusIndex(index_path=index_path)
    loaded.load()
    assert len(loaded.passages) == 10


def test_question_answering_and_cache(tmp_path: Path):
    index_path = tmp_path / "campus_index.json.gz"
    index = CompressedCampusIndex(index_path=index_path)
    index.build(Path("data/course_catalog.json"), Path("data/campus_policies.json"))
    index.load()

    bot = CampusInfoBot(index)

    first = bot.ask("What are the prerequisites for CS 301?")
    assert "CS 301" in json.dumps(first)
    assert first["from_cache"] is False
    assert first["sources"]

    second = bot.ask("What are the prerequisites for CS 301?")
    assert second["from_cache"] is True


def test_unknown_question(tmp_path: Path):
    index_path = tmp_path / "campus_index.json.gz"
    index = CompressedCampusIndex(index_path=index_path)
    index.build(Path("data/course_catalog.json"), Path("data/campus_policies.json"))
    index.load()

    bot = CampusInfoBot(index)
    result = bot.ask("What shuttle color goes to the moon campus?")
    assert "could not find" in result["answer"].lower()
