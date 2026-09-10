"""spaCy-first extraction of BCC target spans and SAM3.1 context prompts.

This module implements the concrete-entity policy from TEXT_ANNOTATION.md:
independent people, objects, body parts, garments, surfaces, and stuff are
linkable; scene/viewpoint words and abstract concepts are not.  The original
surface span and normalized SAM prompt are always kept separately.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Iterable


PERSON_WORDS = {"boy", "child", "girl", "guy", "lady", "man", "person", "woman"}
TARGET_ALIASES = {
    "airplane": {"airplane", "aircraft", "plane"},
    "bear": {"bear", "panda"},
    "bus": {"bus"},
    "cat": {"cat"},
    "dog": {"dog"},
    "fish": {"fish"},
    "giraffe": {"giraffe"},
    "owl": {"owl"},
    "parrot": {"bird", "parrot"},
    "person": PERSON_WORDS | {"diver", "skateboarder", "surfer"},
    "squirrel": {"squirrel"},
    "train": {"train"},
}

# These noun heads do not denote an independently visible entity in the usage
# patterns present in Ref-YT-VOS/ReVOS.  A physical camera remains eligible;
# only a viewpoint camera is filtered by ``_is_viewpoint_camera`` below.
ABSTRACT_OR_META_HEADS = {
    "background",
    "beginning",
    "exercise",
    "family",
    "foreground",
    "front",
    "frame",
    "genus",
    "image",
    "left",
    "matter",
    "motion",
    "order",
    "place",
    "position",
    "race",
    "right",
    "screen",
    "side",
    "strike",
    "subject",
    "thing",  # explicit project exception: unsafe semantic prompt
    "view",
    "video",
}
GENERIC_UNSAFE_HEADS = {"object", "something", "stuff", "thing"}
EVENT_HEADS = {
    "beginning",
    "exercise",
    "game",
    "motion",
    "race",
    "strike",
    "speed",
}
BODY_PARTS = {
    "arm",
    "face",
    "foot",
    "hand",
    "head",
    "leg",
    "mouth",
    "paw",
    "tail",
    "trunk",
    "wing",
}
TAXONOMY_MARKERS = {"family", "genus", "order", "species"}
TARGET_TRAILING_ACTIONS = {
    "hang",
    "hanging",
    "leaning",
    "running",
    "seating",
    "sitting",
    "standing",
    "talking",
    "wearing",
}


@dataclass(frozen=True)
class EntityCandidate:
    candidate_id: str
    role: str
    surface: str
    start: int
    end: int
    head: str
    sam_prompt: str | None
    decision: str
    reason: str
    required_by_bcc: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=1)
def load_spacy():
    import spacy

    pipeline = spacy.load("en_core_web_sm", exclude=["ner"])
    if not pipeline.has_pipe("parser"):
        raise RuntimeError("en_core_web_sm parser is required for BCC extraction")
    return pipeline


def _lemma(token) -> str:
    value = (token.lemma_ or token.text).casefold().strip()
    value = re.sub(r"\((?:e?s)\)?", "", value)
    return re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", value)


def _clean_surface_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _candidate_span(chunk, text: str) -> tuple[int, int]:
    """Keep articles/premodifiers but avoid trailing punctuation."""

    start, end = _clean_surface_span(text, chunk.start_char, chunk.end_char)
    while end > start and text[end - 1] in ",.;:!?\"":
        end -= 1
    # ReVOS uses parenthesized plurality (person(s), hand(s), creature(s)).
    # spaCy usually ends the noun chunk before the closing parenthesis.
    if end < len(text) and text[end] == ")" and re.search(r"\((?:e?s)$", text[start:end].casefold()):
        end += 1
    return start, end


def _prompt_for_chunk(chunk) -> str:
    root = chunk.root
    allowed = {"amod", "compound", "nmod"}
    pieces = [
        _lemma(token)
        for token in chunk
        if token.i <= root.i
        and token.dep_ in allowed
        and token.is_alpha
        and _lemma(token) not in {"first", "last", "left", "right"}
    ]
    pieces.append(_lemma(root))
    prompt = " ".join(dict.fromkeys(piece for piece in pieces if piece))
    prompt = re.sub(r"\bflatscreen\s+t\s+v\b", "television", prompt)
    prompt = re.sub(r"\bt\s+v\b", "television", prompt)
    return prompt


def _is_viewpoint_camera(text: str, start: int, end: int) -> bool:
    nearby = text[max(0, start - 48) : min(len(text), end + 24)].casefold()
    return bool(
        re.search(r"(?:away|distance|far|furthest|closest|nearer|toward|from)\s+(?:the\s+)?camera", nearby)
        or re.search(r"camera\s+(?:view|frame|position)", nearby)
    )


def _taxonomy_context(text: str, start: int, end: int) -> bool:
    prefix = text[max(0, start - 28) : end].casefold()
    return any(re.search(rf"\b{word}\b", prefix) for word in TAXONOMY_MARKERS)


def _quoted_title_context(text: str, start: int, end: int) -> bool:
    before = text[:start]
    through = text[:end]
    return (
        before.count('"') % 2 == 1
        or through.count('"') % 2 == 1
        or before.count("“") > before.count("”")
        or through.count("“") > through.count("”")
    )


def _comparison_context(text: str, start: int) -> bool:
    prefix = text[max(0, start - 18) : start].casefold()
    return bool(re.search(r"\b(?:like|as)\s+(?:an?\s+)?$", prefix))


def _target_chunk(doc, category: str | None):
    chunks = list(doc.noun_chunks)
    # Referring expressions conventionally put the referent in the first NP.
    # Do not let a coarse dataset category redirect us to a later comparison
    # noun (e.g. scooter "driven like a skateboard").
    for chunk in chunks:
        if chunk.root.dep_ in {"nsubj", "nsubjpass", "ROOT", "attr", "dobj"}:
            return chunk
    return chunks[0] if chunks else None


def _who_target(text: str) -> tuple[int, int] | None:
    match = re.match(r"\s*(who)\b", text, flags=re.IGNORECASE)
    return match.span(1) if match else None


def _which_target(text: str) -> tuple[int, int] | None:
    match = re.match(
        r"\s*(Which\b.+?)(?=\s+(?:is|are|was|were|could|would|has|have|does|do)\b)",
        text,
        flags=re.IGNORECASE,
    )
    return _clean_surface_span(text, *match.span(1)) if match else None


def _head_in_span(doc, start: int, end: int, fallback: str) -> str:
    nouns = [
        _lemma(token)
        for token in doc
        if start <= token.idx < end and token.pos_ in {"NOUN", "PROPN"} and _lemma(token)
    ]
    return nouns[-1] if nouns else fallback


def _trim_target_chunk(chunk, text: str) -> tuple[int, int]:
    start, end = _candidate_span(chunk, text)
    tokens = [token for token in chunk if token.idx < end]
    while tokens and (
        _lemma(tokens[-1]) in TARGET_TRAILING_ACTIONS
        or tokens[-1].pos_ in {"VERB", "AUX"}
    ):
        end = tokens[-1].idx
        tokens.pop()
    return _clean_surface_span(text, start, end)


def _make_candidate(
    *,
    index: int,
    role: str,
    text: str,
    start: int,
    end: int,
    head: str,
    prompt: str | None,
    decision: str,
    reason: str,
    required: bool,
) -> EntityCandidate:
    return EntityCandidate(
        candidate_id=f"entity-{index:02d}",
        role=role,
        surface=text[start:end],
        start=start,
        end=end,
        head=head,
        sam_prompt=prompt,
        decision=decision,
        reason=reason,
        required_by_bcc=required,
    )


def extract_entities(
    text: str,
    *,
    target_category: str | None = None,
    negative: bool = False,
) -> dict[str, Any]:
    """Return a target anchor plus contextual candidates and audit decisions."""

    if negative:
        return {
            "spacy_model": "en_core_web_sm",
            "target": None,
            "contexts": [],
            "ignored": [],
            "notes": ["nonexistent-object expression: segmentation intentionally skipped"],
        }
    doc = load_spacy()(text)
    target_chunk = _target_chunk(doc, target_category)
    who = _who_target(text)
    which = _which_target(text)
    if who is not None:
        target_start, target_end = who
        target_head, target_prompt = "person", "person"
    elif which is not None:
        target_start, target_end = which
        target_head = _head_in_span(doc, target_start, target_end, "entity")
        target_prompt = target_head
    elif target_chunk is not None:
        target_start, target_end = _trim_target_chunk(target_chunk, text)
        target_head = _head_in_span(doc, target_start, target_end, _lemma(target_chunk.root))
        target_prompt = target_head
    else:
        # This is recorded rather than silently inventing a referent.
        target_start, target_end = 0, len(text)
        target_head = str(target_category or "entity").casefold()
        target_prompt = target_head
    candidates: list[EntityCandidate] = [
        _make_candidate(
            index=0,
            role="target_gt",
            text=text,
            start=target_start,
            end=target_end,
            head=target_head,
            prompt=target_prompt,
            decision="anchor_ground_truth_tracklet",
            reason="main referring expression is anchored to the dataset tracklet",
            required=True,
        )
    ]

    target_coreference_spans: list[dict[str, Any]] = []
    for token in doc:
        lowered = token.text.casefold()
        if lowered not in {"he", "her", "hers", "him", "his", "its", "she", "that", "their", "them", "they", "who", "whose"}:
            continue
        # Restrict to grammatical references, not determiners such as "that car".
        if token.dep_ not in {"nsubj", "nsubjpass", "poss", "relcl", "attr", "dobj", "pobj"} and lowered not in {"who", "whose", "that"}:
            continue
        if target_start <= token.idx < target_end:
            continue
        target_coreference_spans.append(
            {"start": token.idx, "end": token.idx + len(token.text), "text": token.text}
        )

    seen: set[tuple[int, int, str]] = {(target_start, target_end, target_head)}
    next_index = 1
    for chunk in doc.noun_chunks:
        start, end = _candidate_span(chunk, text)
        head = _lemma(chunk.root)
        if not head or (start >= target_start and end <= target_end) or start <= target_start < end:
            continue
        if head in {"he", "her", "him", "his", "it", "its", "she", "that", "their", "them", "they", "which", "who", "whose"}:
            continue
        key = (start, end, head)
        if key in seen:
            continue
        seen.add(key)
        prompt = _prompt_for_chunk(chunk) or head
        decision = "prompt_sam3.1"
        reason = "independent concrete visual entity required by BCC"
        required = True
        if head in GENERIC_UNSAFE_HEADS:
            decision = "ignore_special_unsafe_generic"
            reason = "project exception: generic thing/object cannot be prompted reliably"
            required = False
            prompt = None
        elif head in EVENT_HEADS or head in ABSTRACT_OR_META_HEADS:
            decision = "ignore_nonvisual_or_meta"
            reason = "abstract event, scene/view, or relational noun is not an entity tracklet"
            required = False
            prompt = None
        elif head == "camera" and _is_viewpoint_camera(text, start, end):
            decision = "ignore_viewpoint_camera"
            reason = "camera denotes the recording viewpoint, not a visible physical camera"
            required = False
            prompt = None
        elif _taxonomy_context(text, start, end):
            decision = "ignore_taxonomic_concept"
            reason = "taxonomic label is linguistic classification, not another visible entity"
            required = False
            prompt = None
        elif _quoted_title_context(text, start, end):
            decision = "ignore_quoted_title"
            reason = "quoted title/language is not another visible entity"
            required = False
            prompt = None
        elif _comparison_context(text, start):
            decision = "ignore_comparison_concept"
            reason = "comparison noun names a class, not another asserted video entity"
            required = False
            prompt = None
        elif re.search(r"\bwidely\s+used\s+for\b|\bused\s+as\b", text[:start], flags=re.IGNORECASE):
            decision = "ignore_generic_definition"
            reason = "definition/use-case language does not assert another visible video entity"
            required = False
            prompt = None
        candidates.append(
            _make_candidate(
                index=next_index,
                role="context_sam" if required else "special_handling",
                text=text,
                start=start,
                end=end,
                head=head,
                prompt=prompt,
                decision=decision,
                reason=reason,
                required=required,
            )
        )
        next_index += 1

        # spaCy noun chunks include a possessive inside the possessed entity.
        # BCC needs both tracks: "the person's hand" -> person and hand, with
        # overlapping text spans allowed by the annotation rules.
        for token in chunk:
            if token.dep_ != "poss" or token.pos_ not in {"NOUN", "PROPN"}:
                continue
            poss_start = token.idx
            for left in token.lefts:
                if left.dep_ == "det":
                    poss_start = min(poss_start, left.idx)
            poss_end = token.idx + len(token.text)
            poss_head = _lemma(token)
            poss_key = (poss_start, poss_end, poss_head)
            if poss_key in seen or (poss_start >= target_start and poss_end <= target_end):
                continue
            seen.add(poss_key)
            candidates.append(
                _make_candidate(
                    index=next_index,
                    role="context_sam",
                    text=text,
                    start=poss_start,
                    end=poss_end,
                    head=poss_head,
                    prompt=poss_head,
                    decision="prompt_sam3.1",
                    reason="possessor is a separately visible entity; overlap is valid BCC",
                    required=True,
                )
            )
            next_index += 1

    # Body parts can be nested or tokenized outside a usable noun chunk in the
    # parenthesized ReVOS grammar (hand(s)); add conservative token fallbacks.
    covered = [(candidate.start, candidate.end, candidate.head) for candidate in candidates]
    for token in doc:
        head = _lemma(token)
        if head not in BODY_PARTS:
            continue
        if any(start <= token.idx < end and prior_head == head for start, end, prior_head in covered):
            continue
        start, end = token.idx, token.idx + len(token.text)
        candidates.append(
            _make_candidate(
                index=next_index,
                role="context_sam",
                text=text,
                start=start,
                end=end,
                head=head,
                prompt=head,
                decision="prompt_sam3.1",
                reason="body parts are independent BCC entities",
                required=True,
            )
        )
        next_index += 1

    return {
        "spacy_model": "en_core_web_sm",
        "target": candidates[0].as_dict(),
        "target_coreference_spans": target_coreference_spans,
        "contexts": [
            candidate.as_dict() for candidate in candidates[1:] if candidate.required_by_bcc
        ],
        "ignored": [
            candidate.as_dict() for candidate in candidates[1:] if not candidate.required_by_bcc
        ],
        "notes": [],
    }


def grouped_prompts(extraction: dict[str, Any]) -> list[dict[str, Any]]:
    """Deduplicate SAM calls while preserving every exact linked text span."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in extraction.get("contexts", []):
        prompt = str(candidate.get("sam_prompt") or "").strip().casefold()
        if prompt:
            grouped.setdefault(prompt, []).append(candidate)
    return [
        {"sam_prompt": prompt, "candidates": candidates}
        for prompt, candidates in sorted(grouped.items())
    ]
