---
name: Write scripts to files instead of inline code
description: User prefers writing code to files and running them, not inline python -c commands in bash
type: feedback
---

Don't run inline Python via `python -c "..."` in bash. Write to a file first, then run it.

**Why:** User finds inline code harder to review and modify.

**How to apply:** Always write scripts to a file (e.g., `scripts/foo.py`) and then run `python scripts/foo.py`.
