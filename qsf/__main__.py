"""Entrypoint bridge for ``python -m qsf`` from repository root."""

import sys

from qsf.cli import main


if __name__ == "__main__":
    sys.exit(main())
