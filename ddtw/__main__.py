"""Allow the CLI to use the active interpreter: python -m ddtw."""

from .cli import main

raise SystemExit(main())
