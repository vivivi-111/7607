from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Callable


@dataclass(frozen=True)
class Extraction:
    value: str | None
    status: str  # ANSWER, NONE, or HEDGE


FINAL_PATTERNS = [
    re.compile(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:\*{1,2})?final\s+answer"
        r"(?:\*{1,2})?\s*[:\-]\s*(.+?)\s*$"
    ),
    re.compile(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:\*{1,2})?answer"
        r"(?:\*{1,2})?\s*[:\-]\s*(.+?)\s*$"
    ),
    re.compile(r"(?i)\bfinal\s+answer\s*[:\-]\s*(.+?)(?:\n|$)"),
    re.compile(
        r"(?i)\b(?:the\s+)?(?:correct\s+)?(?:answer|choice)"
        r"(?:\s+provided)?\s+"
        r"(?:is|was|would\s+be|should\s+be)\s*:?\s*"
        r"(.+?)(?:[.!]\s|$)"
    ),
    re.compile(r"\\boxed\{([^{}]+)\}"),
]
HEDGE_PATTERN = re.compile(
    r"(?i)\b(?:cannot\s+(?:be\s+)?determin\w*|not enough information|"
    r"unsure|uncertain|it depends)\b"
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "which",
    "with",
    "would",
}
NEGATION_TOKENS = {"false", "incorrect", "myth", "no", "not", "wrong"}
NUMBER_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}
COMPOUND_MEASUREMENT_PATTERN = re.compile(
    r"(?i)[-+]?\d+(?:\.\d+)?\s*"
    r"(?:feet|foot|inches?|hours?|minutes?|seconds?)"
    r"(?:\s+and)?\s+[-+]?\d+(?:\.\d+)?\s*"
    r"(?:feet|foot|inches?|hours?|minutes?|seconds?)"
)
ECHOED_INSTRUCTION_PATTERN = re.compile(
    r"(?is)\bbriefly reassess and end with\s+[`'\"]*final answer:"
    r".*?\bon its own line[.`'\"]*\s*$"
)


def normalize_number(value: str) -> str | None:
    cleaned = (
        value.strip()
        .replace("\\", "")
        .replace(",", "")
        .replace("$", "")
        .replace("%", "")
        .rstrip(".")
    )
    cleaned = re.sub(r"(?i)\s*(?:dollars?|units?)\s*$", "", cleaned)
    match = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not match:
        numbers = re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?", value.replace("\\", ""))
        if len(numbers) != 1:
            return None
        cleaned = numbers[0].replace(",", "").replace("$", "")
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    normalized = format(number.normalize(), "f")
    return "0" if Decimal(normalized) == 0 else normalized


def _candidates(text: str) -> list[str]:
    matches: list[str] = []
    for pattern in FINAL_PATTERNS:
        for match in pattern.finditer(text):
            matches.append(match.group(1).strip().strip("*_` "))
    if not matches:
        standalone = re.search(r"(?im)^\s*\(?([A-Z])\)?[.)]?\s*$", text)
        if standalone:
            matches.append(standalone.group(1))
    return matches


def _normalized_text(value: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(value.casefold()))


def _tokens(value: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(value.casefold())) - STOPWORDS


def _choice_from_text(candidate: str, choices: list[str]) -> str | None:
    cleaned = candidate.strip().rstrip(".")
    normalized_candidate = _normalized_text(cleaned)
    normalized_choices = [_normalized_text(str(choice)) for choice in choices]
    exact = [
        index
        for index, normalized in enumerate(normalized_choices)
        if normalized == normalized_candidate
    ]
    if len(exact) == 1:
        return chr(65 + exact[0])
    contained = [
        index
        for index, normalized in enumerate(normalized_choices)
        if normalized
        and (
            normalized in normalized_candidate
            or (
                len(normalized_candidate) >= 5
                and normalized_candidate in normalized
            )
        )
    ]
    if len(contained) == 1:
        return chr(65 + contained[0])
    candidate_negated = "not" in _tokens(candidate)
    similarities = []
    for choice, normalized in zip(choices, normalized_choices, strict=True):
        choice_negated = "not" in _tokens(str(choice))
        similarities.append(
            0.0
            if candidate_negated != choice_negated
            else difflib.SequenceMatcher(
                None, normalized_candidate, normalized
            ).ratio()
        )
    ranked = sorted(
        range(len(similarities)), key=similarities.__getitem__, reverse=True
    )
    if (
        ranked
        and similarities[ranked[0]] >= 0.72
        and (
            len(ranked) == 1
            or similarities[ranked[0]] - similarities[ranked[1]] >= 0.08
        )
    ):
        return chr(65 + ranked[0])
    return None


def _choice_from_response(text: str, choices: list[str]) -> str | None:
    normalized = _normalized_text(text)
    single_token_counts = []
    for choice in choices:
        tokens = list(_tokens(str(choice)))
        if len(tokens) == 1:
            single_token_counts.append(
                len(re.findall(rf"\b{re.escape(tokens[0])}\b", normalized))
            )
        else:
            single_token_counts.append(0)
    ranked_counts = sorted(
        range(len(choices)), key=single_token_counts.__getitem__, reverse=True
    )
    if (
        single_token_counts[ranked_counts[0]] >= 2
        and (
            len(ranked_counts) == 1
            or single_token_counts[ranked_counts[0]]
            > single_token_counts[ranked_counts[1]]
        )
    ):
        return chr(65 + ranked_counts[0])

    response_tokens = _tokens(text)
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    scores: list[float] = []
    for choice in choices:
        choice_tokens = _tokens(str(choice))
        if not choice_tokens:
            scores.append(0.0)
            continue
        coverage = len(choice_tokens & response_tokens) / len(choice_tokens)
        best_sentence_score = 0.0
        best_sentence_negated = False
        for sentence in sentences:
            sentence_tokens = _tokens(sentence)
            union = choice_tokens | sentence_tokens
            similarity = (
                len(choice_tokens & sentence_tokens) / len(union)
                if union
                else 0.0
            )
            if similarity > best_sentence_score:
                best_sentence_score = similarity
                best_sentence_negated = bool(
                    sentence_tokens & NEGATION_TOKENS
                ) and not bool(choice_tokens & NEGATION_TOKENS)
        score = 0.6 * coverage + 0.4 * best_sentence_score
        if best_sentence_negated:
            score *= 0.35
        scores.append(score)
    ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    if (
        ranked
        and scores[ranked[0]] >= 0.4
        and (
            len(ranked) == 1
            or scores[ranked[0]] - scores[ranked[1]] >= 0.06
        )
    ):
        return chr(65 + ranked[0])
    return None


def extract_answer(
    text: str,
    choices: list[str] | None = None,
    llm_parser: Callable[[str, list[str] | None], str | None] | None = None,
) -> Extraction:
    if ECHOED_INSTRUCTION_PATTERN.search(text):
        return Extraction(None, "NONE")
    candidates = _candidates(text)
    saw_multiple_answers = False
    saw_unmapped_explicit_answer = False
    for candidate in candidates:
        if choices is None:
            rounded = re.search(
                r"(?i)\bround(?:ed|ing)?\s+(?:it\s+)?(?:up|down)\s+to\s+"
                r"([-+]?\$?\d[\d,]*(?:\.\d+)?)",
                candidate,
            )
            if rounded:
                return Extraction(
                    normalize_number(rounded.group(1)), "ANSWER"
                )
            if COMPOUND_MEASUREMENT_PATTERN.search(candidate):
                continue
            number = normalize_number(candidate)
            if number is not None:
                return Extraction(number, "ANSWER")
            leading_number = re.match(
                r"^\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)\b", candidate
            )
            if leading_number:
                number = normalize_number(leading_number.group(1))
                if number is not None:
                    return Extraction(number, "ANSWER")
            continue
        if re.match(
            r"(?i)^\s*\(?[A-Z]\)?\s+(?:is|was)\s+(?:in)?correct\b",
            candidate,
        ):
            continue
        if re.match(
            r"^\s*\(?[A-Z]\)?\s*(?:,|and|&)\s*\(?[A-Z]\)?",
            candidate,
        ):
            saw_multiple_answers = True
            continue
        label_match = re.match(
            r"^\(?([A-Z])\)?(?:[.):]|,\s+(?![A-Z](?:\W|$))|$)",
            candidate.strip(),
        )
        if label_match:
            index = ord(label_match.group(1)) - ord("A")
            if 0 <= index < len(choices):
                return Extraction(chr(65 + index), "ANSWER")
        mentioned_labels = {
            label
            for label in re.findall(
                r"(?<![A-Za-z0-9])([A-Z])(?![A-Za-z0-9])", candidate
            )
            if ord(label) - ord("A") < len(choices)
        }
        if len(mentioned_labels) > 1:
            if re.search(
                r"(?i)\ball (?:of )?(?:the )?(?:choices|options)\b"
                r".*?\b(?:accepted|correct|true)\b",
                candidate,
            ):
                saw_unmapped_explicit_answer = True
                continue
            saw_multiple_answers = True
            continue
        numeric_candidate = normalize_number(candidate)
        if numeric_candidate in NUMBER_WORDS:
            matching = [
                index
                for index, choice in enumerate(choices)
                if _normalized_text(str(choice))
                == NUMBER_WORDS[numeric_candidate]
            ]
            if len(matching) == 1:
                return Extraction(chr(65 + matching[0]), "ANSWER")
        inferred = _choice_from_text(candidate, choices)
        if inferred is not None:
            return Extraction(inferred, "ANSWER")
        saw_unmapped_explicit_answer = True
    if saw_multiple_answers:
        return Extraction(None, "HEDGE")
    if HEDGE_PATTERN.search(text):
        return Extraction(None, "HEDGE")
    if saw_unmapped_explicit_answer:
        return Extraction(None, "NONE")
    if choices:
        nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if nonempty_lines:
            last_label = re.match(r"^\(?([A-Z])\)?[.)]\s+\S", nonempty_lines[-1])
            if last_label:
                index = ord(last_label.group(1)) - ord("A")
                if 0 <= index < len(choices):
                    return Extraction(chr(65 + index), "ANSWER")
        tail = " ".join(text[-600:].casefold().split())
        choice_hits = [
            index
            for index, choice in enumerate(choices)
            if (
                normalized_choice := " ".join(
                    str(choice).strip().rstrip(".").casefold().split()
                )
            )
            and (
                len(normalized_choice) >= 3
                or re.fullmatch(r"[-+]?\d+(?:\.\d+)?", normalized_choice)
            )
            and normalized_choice in tail
        ]
        if len(choice_hits) == 1:
            return Extraction(chr(65 + choice_hits[0]), "ANSWER")
        if text.rstrip().endswith((".", "!", "?")):
            inferred = _choice_from_response(text, choices)
            if inferred is not None:
                return Extraction(inferred, "ANSWER")
    else:
        final_sentence = re.search(
            r"(?is)(?:therefore|thus|so)[^.\n]*?([-$]?\d[\d,]*(?:\.\d+)?)"
            r"(?:\s+[a-z]+)?\.?\s*$",
            text,
        )
        if final_sentence:
            number = normalize_number(final_sentence.group(1))
            if number is not None:
                return Extraction(number, "ANSWER")
        numbers = {
            normalized
            for value in re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?", text)
            if (normalized := normalize_number(value)) is not None
        }
        if len(numbers) == 1 and re.search(
            r"(?i)\b(?:answer|total|price|equals?|indeed)\b", text
        ):
            return Extraction(numbers.pop(), "ANSWER")
    if llm_parser is not None:
        parsed = llm_parser(text, choices)
        if parsed is not None:
            return Extraction(parsed, "ANSWER")
    return Extraction(None, "NONE")


def expected_value(record: dict) -> str:
    if record.get("choices"):
        return chr(65 + int(record["gold_index"]))
    normalized = normalize_number(str(record["gold_answer"]))
    if normalized is None:
        raise ValueError("gold answer is not numeric")
    return normalized


def wrong_target_value(record: dict) -> str:
    if record.get("choices"):
        return chr(65 + int(record["wrong_target_index"]))
    normalized = normalize_number(str(record["wrong_target"]))
    if normalized is None:
        raise ValueError("wrong target is not numeric")
    return normalized
