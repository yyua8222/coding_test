"""Demonstrate filtering and two-pass rewriting of a deliberately difficult story."""

from src.solution import WORD_PATTERN, filter, generate

HARD_STORY = (
    "Mira's iridescent, translucent amulet shimmered beneath the moon—"
    "until a mischievous dragon whispered, “Abracadabra!” 😀"
)


def remove_flagged_words(story: str, flagged_words: list[str]) -> str:
    rejected = set(flagged_words)
    return WORD_PATTERN.sub(
        lambda match: "" if match.group(0).lower() in rejected else match.group(0),
        story,
    )


def main() -> None:
    filtered_story, flagged_words = filter(HARD_STORY)
    without_flagged_words = remove_flagged_words(filtered_story, flagged_words)
    checked_story, remaining_words = filter(without_flagged_words)

    print("Hard input story:")
    print(HARD_STORY)
    print("\nWords flagged by filter:")
    print(", ".join(sorted(set(flagged_words))) or "none")
    print("\nAfter filter removes forbidden characters and the demo removes flagged words:")
    print(without_flagged_words)

    if checked_story != without_flagged_words:
        raise RuntimeError("Disallowed characters remain after filtering")
    if remaining_words:
        raise RuntimeError(f"Flagged words remain after removal: {remaining_words}")

    request = (
        "Rewrite this hard story in simple English. Keep the main idea and give it "
        f"a kind ending.\n\n{HARD_STORY}"
    )
    final_story = generate(request, max_new_tokens=120)
    final_story, final_flagged_words = filter(final_story)
    if final_flagged_words:
        raise RuntimeError(
            f"Final story still has words outside the vocabulary: {final_flagged_words}"
        )

    print("\nFinal Qwen output-2:")
    print(final_story)
    print("\nFinal filter check: passed")


if __name__ == "__main__":
    main()
