"""Implement `generate()` here. See CANDIDATE_TASK.md for the spec."""

import re
import string
from functools import lru_cache
from pathlib import Path

MODEL_NAME = "Qwen/Qwen3-0.6B"
WORDLIST_PATH = Path(__file__).resolve().parent / "oxford3000_tokens_normalized.txt"
USER_PROMPT_PATH = Path(__file__).resolve().parent.parent / "user_prompt.txt"
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
ALLOWED_WHITESPACE = {" ", "\t", "\r", "\n"}
MAX_REWRITE_ATTEMPTS = 3


@lru_cache(maxsize=1)
def _load_vocabulary() -> frozenset[str]:
    entries = WORDLIST_PATH.read_text(encoding="utf-8").lower().splitlines()
    vocabulary: set[str] = set()
    for entry in entries:
        vocabulary.update(entry.split())
    return frozenset(vocabulary)


def filter(story: str) -> tuple[str, list[str]]:
    """Remove disallowed characters and return the story's unmatched words."""
    cleaned_story = "".join(
        char
        for char in story
        if char in ALLOWED_WHITESPACE
        or (
            char.isascii()
            and (char.isalnum() or char in string.punctuation)
        )
    )
    vocabulary = _load_vocabulary()
    unmatched_words = [
        match.group(0).lower()
        for match in WORD_PATTERN.finditer(cleaned_story)
        if match.group(0).lower() not in vocabulary
    ]
    return cleaned_story, unmatched_words


@lru_cache(maxsize=1)
def _load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
    model.eval()
    return tokenizer, model


def _load_user_prompt() -> str:
    for line in USER_PROMPT_PATH.read_text(encoding="utf-8").splitlines():
        prompt = re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
        if prompt:
            return prompt
    raise ValueError(f"No prompt found in {USER_PROMPT_PATH}")


def generate(prompt: str | None = None, max_new_tokens: int = 60) -> str:
    """Generate and filter a story from prompt, or user_prompt.txt by default."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")

    vocabulary = _load_vocabulary()
    user_prompt = prompt.strip() if prompt and prompt.strip() else _load_user_prompt()
    tokenizer, model = _load_model()
    vocabulary_text = ", ".join(sorted(vocabulary))
    writing_rules = (
        "Write or rewrite a short story in response to the user's request. Use "
        "simple English and only Oxford 3000 words. Replace unfamiliar words with "
        "simple words or rewrite the full sentence when needed. Compare words "
        "case-insensitively; do not use contractions. Use only ASCII letters, "
        "digits, ordinary ASCII punctuation, and space, tab, carriage return, or "
        "newline. Return only the story, with no explanation or title."
    )
    fast_system_prompt = writing_rules
    structured_system_prompt = (
        writing_rules
        + " Follow the supplied repair fields exactly. Every word in the final "
        "story must be in the supplied Oxford 3000 list."
    )
    full_dictionary_system_prompt = (
        writing_rules + f"\n\nAllowed Oxford 3000 words: {vocabulary_text}"
    )

    def run_generation(user_content: str, system_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"] if not hasattr(encoded, "shape") else encoded

        import torch

        with torch.inference_mode():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        return tokenizer.decode(
            output[0, input_ids.shape[-1]:],
            skip_special_tokens=True,
        ).strip()

    draft = run_generation(
        f"User request:\n{user_prompt}\nWrite a short story that follows this request.",
        fast_system_prompt,
    )

    def make_feedback(words: list[str]) -> str:
        return (
            "Replace every rejected word below with an easier word from the "
            "allowed vocabulary. Do not repeat any rejected word. If needed, "
            "rewrite the full sentence in simple English. Rejected words: "
            f"{', '.join(sorted(set(words)))}."
        )

    for attempt in range(MAX_REWRITE_ATTEMPTS + 1):
        draft, unmatched_words = filter(draft)
        if not unmatched_words:
            return draft
        if attempt == MAX_REWRITE_ATTEMPTS:
            break

        draft = run_generation(
            "Rewrite the latest story draft as a simple story. Replace every "
            "rejected word with an easier Oxford 3000 word, or rewrite its "
            "sentence if needed. Do not use any rejected word. "
            f"This is rewrite attempt {attempt + 1} of {MAX_REWRITE_ATTEMPTS}.\n"
            f"User request:\n{user_prompt}\n"
            f"Filter feedback:\n{make_feedback(unmatched_words)}\n"
            f"Latest story draft:\n{draft}",
            fast_system_prompt,
        )

    unmatched_text = ", ".join(sorted(set(unmatched_words)))
    draft = run_generation(
        "Repair the story using this structured input. Replace each rejected "
        "word with the best-fitting word from the allowed list; rewrite a full "
        "sentence if necessary. Return the complete corrected story only.\n"
        f"USER_REQUEST:\n{user_prompt}\n"
        f"REJECTED_WORDS:\n{unmatched_text}\n"
        f"STORY_DRAFT:\n{draft}\n"
        f"ALLOWED_OXFORD_3000_WORDS:\n{vocabulary_text}",
        structured_system_prompt,
    )
    draft, unmatched_words = filter(draft)
    if not unmatched_words:
        return draft

    draft = run_generation(
        "Rewrite the complete story using only words from the Oxford 3000 list "
        "in the system instructions. Replace the rejected words or rewrite their "
        "sentences while preserving the story's meaning.\n"
        f"User request:\n{user_prompt}\n"
        f"Rejected words:\n{', '.join(sorted(set(unmatched_words)))}\n"
        f"Latest story draft:\n{draft}",
        full_dictionary_system_prompt,
    )
    draft, unmatched_words = filter(draft)
    if not unmatched_words:
        return draft

    raise RuntimeError(
        "Qwen could not produce a story that satisfies the vocabulary rules "
        "after fast, structured, and full-dictionary repairs. "
        f"Unmatched words: {', '.join(sorted(set(unmatched_words)))}"
    )
