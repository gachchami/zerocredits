"""Run the application once against a task file.

This lightweight wrapper exists for local evaluation and intentionally delegates to
main.py so benchmark runs exercise the same production path.
"""
import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    subprocess.run(
        [sys.executable, "main.py", "--input", args.input, "--output", args.output],
        check=True,
    )


if __name__ == "__main__":
    main()
