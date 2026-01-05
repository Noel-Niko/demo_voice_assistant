#!/usr/bin/env python

import sys

if __name__ == "__main__":
    print(
        "Whisper tests have been retired. This repository now uses Google Cloud "
        "Speech-to-Text v2 only.\n"
        "Run the full test suite with: uv run pytest tests"
    )
    sys.exit(0)
