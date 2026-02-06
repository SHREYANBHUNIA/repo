# Campus Information Bot

A lightweight campus information bot that ingests course catalog and campus policy content, compresses it, and answers student questions using fast local retrieval.

## Why this design lowers operational cost

- **Compressed corpus storage** (`gzip` + compact JSON) reduces storage and memory footprint.
- **Keyword-based retrieval** avoids expensive model inference for many FAQ-style questions.
- **Short context assembly** returns only top relevant snippets to keep prompt/context size small when integrating with LLMs.
- **Answer caching** prevents repeated compute for common student questions.

## Quick start

```bash
python3 campus_bot.py --build-index
python3 campus_bot.py --ask "What are the prerequisites for CS 301?"
```

## Files

- `campus_bot.py`: Bot implementation and CLI.
- `data/course_catalog.json`: Sample compressed-friendly course catalog source.
- `data/campus_policies.json`: Sample campus policy source.
- `tests/test_campus_bot.py`: Unit tests.

## Example

```bash
python3 campus_bot.py --ask "How many credits can I take in summer?"
```

Output includes:
- concise answer
- source snippets used
- retrieval latency
- estimated context-token savings from compression
