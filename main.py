"""Entry point for braking automation with status output."""
from __future__ import annotations

from braking_controller import run_braking_loop


def main() -> None:
    run_braking_loop(show_status=True)


if __name__ == "__main__":
    main()
