"""Run the prompts in hard_prompts.txt through the constrained generator."""

from pathlib import Path

from src.solution import filter, generate

PROMPTS_PATH = Path(__file__).resolve().parent / "hard_prompts.txt"


def load_prompts() -> list[str]:
    """Load non-empty prompt lines from the sample file."""
    prompts = [
        line.strip()
        for line in PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError(f"No prompts found in {PROMPTS_PATH}")
    return prompts


def main() -> None:
    for index, prompt in enumerate(load_prompts(), start=1):
        _, unmatched_input_words = filter(prompt)
        print(f"Case {index}: {prompt}")
        print(
            "Unmatched input words: "
            f"{', '.join(sorted(set(unmatched_input_words))) or 'none'}"
        )

        output = generate(prompt, max_new_tokens=60)
        output, unmatched_output_words = filter(output)
        if unmatched_output_words:
            raise RuntimeError(
                f"Case {index} produced out-of-vocabulary words: "
                f"{', '.join(sorted(set(unmatched_output_words)))}"
            )

        print(f"Generated story:\n{output}")
        print("Output check: passed\n")


if __name__ == "__main__":
    main()
