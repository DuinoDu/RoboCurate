# Contributing

Public contribution and publication begin after the items in `LICENSE_STATUS.md` are resolved.

Use Python 3.10+, the application dependencies and pytest. Model work additionally uses a separate environment from `requirements-warp.txt` and a suitable PyTorch installation. Frontend helpers use native ES modules; `node --test tests/*.test.mjs` needs no package download.

Use `scripts/create_demo.py` or synthetic pytest fixtures for reproducible tests. Do not commit recordings, personal information, screenshots of private collections, credentials, generated caches or databases. Describe the data source and task scope when reporting model results; stage accuracy is not final-success accuracy or training benefit.

For a change, explain the affected behavior and verification. Keep raw sources read-only, preserve nanosecond precision, avoid future-frame/action leakage, and keep human review saves explicit. API changes should include schema and reader compatibility notes.
