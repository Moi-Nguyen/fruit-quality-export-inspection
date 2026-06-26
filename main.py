"""Entry point for the fruit quality inspection project."""

from __future__ import annotations

import argparse

from src.dataset_sampling import check_datasets, create_sample_dataset

STARTUP_MESSAGE = "Fruit Quality Inspection project skeleton is ready."


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for project utility commands."""
    parser = argparse.ArgumentParser(description="Fruit quality inspection project utilities.")
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Create a balanced sampled dataset under data/sample/.",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Print class image counts for data/raw/ and data/sample/.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the requested project utility command or print the startup message."""
    args = parse_args()

    if args.sample_data:
        create_sample_dataset()
        return

    if args.check_data:
        check_datasets()
        return

    print(STARTUP_MESSAGE)


if __name__ == "__main__":
    main()
