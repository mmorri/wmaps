import argparse
import os
import subprocess
import sys


def cmd_validate(_: argparse.Namespace) -> int:
    from analyze_json import main as analyze_main

    analyze_main()
    return 0


def cmd_filter(_: argparse.Namespace) -> int:
    from filter_products import main as filter_main

    filter_main()
    return 0


def cmd_verify(_: argparse.Namespace) -> int:
    from verify_filtering import main as verify_main

    verify_main()
    return 0


def cmd_train(_: argparse.Namespace) -> int:
    from train_and_export import train_and_export
    from pathlib import Path

    train_and_export(Path("."))
    return 0


def cmd_api(args: argparse.Namespace) -> int:
    env = os.environ.copy()
    if args.port:
        env["API_PORT"] = str(args.port)
    return subprocess.call([sys.executable, "book_api.py"], env=env)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="wmaps utility CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("validate", help="Validate and analyze JSON files").set_defaults(func=cmd_validate)
    sub.add_parser("filter", help="Filter products with invalid sales ranks").set_defaults(func=cmd_filter)
    sub.add_parser("verify", help="Verify filtering results").set_defaults(func=cmd_verify)
    sub.add_parser("train", help="Train models and export artifacts").set_defaults(func=cmd_train)

    api = sub.add_parser("api", help="Run FastAPI server")
    api.add_argument("--port", type=int, default=8001, help="Port to bind")
    api.set_defaults(func=cmd_api)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

