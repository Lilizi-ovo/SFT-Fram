"""Portable entry point; the control interface uses only the Python standard library."""

from scr.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
