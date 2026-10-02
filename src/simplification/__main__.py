"""
Pkg 3: Simplification command line [last updated 2026-10-02]

Usage:
    python -m src.simplification data/email_recs.xlsx
    python -m src.simplification data/email_recs.xlsx -o data/out.json
    python -m src.simplification in.csv --prompt summarise --text-column body
    python -m src.simplification data/email_recs.xlsx -o data/cleaned.csv --layout long

    The last form writes what the dashboard and the feedback form read.

Resumption:
    This is only argument parsing around core.simplify_file().
"""

import argparse
import sys
import warnings

from .core import MultiplePromptsWarning, resolve_prompts, simplify_file
from .io import SUPPORTED_FORMATS


def _print_progress(done: int, total: int) -> None:
    print(f"\r  {done}/{total} simplifications", end="" if done < total else "\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.simplification",
        description="Simplify the text column of a data file. Keeps every original column "
                    f"and adds the simplification(s). Formats: {', '.join(SUPPORTED_FORMATS)}.",
    )
    parser.add_argument("input", help="input data file")
    parser.add_argument("-o", "--output", help="output file (default: <input>.simplified.<ext>)")
    parser.add_argument("--prompt", action="append", metavar="NAME",
                        help="prompt to use instead of the selected one(s); repeat for several")
    parser.add_argument("--text-column", help="column holding the text (default: auto-detect)")
    parser.add_argument("--model", help="OpenAI model (default: from settings)")
    parser.add_argument("--layout", choices=("wide", "long"), default="wide",
                        help="wide: one column per prompt (default). long: one row per prompt, "
                             "for the dashboard and feedback form")
    parser.add_argument("--sample", type=int, metavar="N", help="only simplify N random rows")
    parser.add_argument("--seed", type=int, default=42, help="random seed for --sample")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="don't ask before running several prompts")
    args = parser.parse_args()

    try:
        prompts = resolve_prompts(args.prompt)
    except (ValueError, KeyError) as e:
        sys.exit(f"Error: {e}")

    if len(prompts) > 1:
        print(f"Warning: {len(prompts)} prompts are selected ({', '.join(prompts)}). "
              f"Every row will get {len(prompts)} simplifications.")
        if not args.yes:
            try:
                answer = input("Continue? [y/N] ")
            except EOFError:
                answer = ""
            if answer.strip().lower() not in ("y", "yes"):
                sys.exit("Stopped. Select one prompt, or pass --yes to run them all.")

    print(f"Simplifying {args.input} with: {', '.join(prompts)}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MultiplePromptsWarning)  # already shown above
        try:
            output = simplify_file(
                args.input, args.output, sample_size=args.sample, seed=args.seed,
                prompts=prompts, text_column=args.text_column, model=args.model,
                layout=args.layout, on_progress=_print_progress,
            )
        except (ValueError, FileNotFoundError) as e:
            sys.exit(f"Error: {e}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
