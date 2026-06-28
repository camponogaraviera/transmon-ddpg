import argparse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reinforce_transmon",
        description="Run Reinforce Transmon training.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Save plot of energy levels during training.",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run inference (no training).",
    )
    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    from reinforce_transmon.src import main as transmon_main

    transmon_main.main(
        render=args.render,
        inference=args.inference,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
