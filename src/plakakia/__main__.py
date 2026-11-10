"""Allow ``python -m plakakia``."""

import sys

from .cli import main

sys.exit(main())
