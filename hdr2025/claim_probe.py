"""
Step 0 of g13, unwired: find superlative and rank claims in the RAW DOCUMENT
rather than in whatever the readers happened to volunteer.

Run against the parks book and against the two header-absent documents before
anything is wired, and check by hand that the entity attached to each claim is
the right one. A claim bound to the wrong entity is worse than a missing claim:
it manufactures a contradiction between two facts that have nothing to do with
each other, which is exactly what the last run did in prose.

    python3 -u claim_probe.py
"""

import collections
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("G12_FIXES", "headers,arcsection,entityseed,provwindow")
_spec = importlib.util.spec_from_file_location(
    "base", os.path.join(HERE, os.environ.get("BASE_PIPELINE", "pipeline_g12.py")))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


# --------------------------------------------------------------- STEP 0

_ORDINAL = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}

# The superlatives check_claims already knows how to adjudicate.
_SUPER_WORD = (r"largest|biggest|highest|tallest|longest|deepest|oldest|"
               r"greatest|smallest|shortest|lowest|widest|most")

# What is being ranked. Deliberately generic nouns -- the words any document
# uses for the thing it is comparing.
_THING = r"[a-z][\w-]*(?:\s+[a-z][\w-]*){0,3}"

# Two shapes, and between them they cover every superlative sentence in the
# documents with score history:
#
#   A  possessive scope   "Norway's largest national park"
#                         "Britain's second-highest peak"
#                         "Austria's highest peak"
#   B  prepositional      "the highest mountain in Wales"
#                         "the largest national park in the Alps"
#                         "Europe's deepest river canyon"  (A, with a 2-word thing)
#
# The ordinal is optional and captured, because a RANK claim is the thing the
# value ledger structurally cannot see: "second-highest" is not a value that
# disagrees with another value, it is an assertion about an ordering.
_CLAIM_A = re.compile(
    rf"\b(?P<scope>[A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)?)['’]s\s+"
    rf"(?:(?P<ordinal>{'|'.join(_ORDINAL)})[-\s]+)?"
    rf"(?P<super>{_SUPER_WORD})\s+"
    rf"(?P<thing>{_THING})", re.I)

_CLAIM_B = re.compile(
    rf"\bthe\s+"
    rf"(?:(?P<ordinal>{'|'.join(_ORDINAL)})[-\s]+)?"
    rf"(?P<super>{_SUPER_WORD})\s+"
    rf"(?P<thing>{_THING}?)\s+"
    rf"(?:in|of)\s+(?:the\s+)?(?P<scope>[A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)?)")

# A superlative inside these is the document describing a category, not making
# a checkable assertion about one entity.
_CLAIM_NOISE = re.compile(r"\b(?:one of|among|some of|few of|many of)\s+the\s*$", re.I)


def find_text_claims(text: str, names: list[str], limit: int = 400) -> list[dict]:
    """
    Superlative and rank claims read out of the document itself.

    THE REASON THIS EXISTS. check_claims() compares a CLAIM line against the
    recorded figures, and CLAIM lines come from whatever a reader chose to
    write down. Measured on the last scored run, that is not enough: the book
    says "the high plateau of Hardangervidda that dominates NORWAY'S LARGEST
    NATIONAL PARK" and "At 1085m, BRITAIN'S SECOND-HIGHEST PEAK has been a
    testing ground since 1798", and no reader emitted either. What they emitted
    instead was "northern Europe's most accessible wilderness areas" and, six
    times over, "the highest mountain in Wales". Both records needed to refute
    those claims were present, verified, and correct -- there was simply no
    claim to refute, so the two contradiction questions had no candidate at all
    and the synthesis model free-formed an answer from the reader lines.

    This is the same move find_label_unit_outliers() already makes for units,
    and for the same reason: reading the raw text makes the detector immune to
    what any reader chose to report. It costs no model call.

    The entity is the nearest preceding name the readers DID return, which is
    how find_label_unit_outliers attributes its sites. Names still come from
    the records; only the claim comes from the text.
    """
    out, seen = [], set()
    for pattern in (_CLAIM_A, _CLAIM_B):
        for m in pattern.finditer(text):
            if _CLAIM_NOISE.search(text[max(0, m.start() - 24):m.start()]):
                continue
            scope = m.group("scope")
            if not scope or scope.lower() in P._NOT_A_SCOPE:
                continue
            claim = " ".join(m.group(0).split())
            entity = P._nearest_name(text, m.start(), names)
            if entity.startswith("("):
                continue                       # no entity to attach it to
            key = (P._entity_key(entity), claim.lower())
            if key in seen:
                continue
            seen.add(key)
            ordinal = (m.groupdict().get("ordinal") or "").lower()
            out.append({
                "entity": entity, "claim": claim,
                "where": "stated in the document text",
                "source": "document text", "from_text": True,
                "superlative": m.group("super").lower(),
                "claimed_rank": _ORDINAL.get(ordinal),
            })
            if len(out) >= limit:
                return out
    return out


# ---------------------------------------------------------------- PROBE

def probe(doc_path: str, submission: str | None, qids: tuple) -> None:
    import json
    text = P.normalize_text(open(doc_path, encoding="utf-8", errors="replace").read())
    body = P.strip_back_matter(text)

    names: list[str] = []
    if submission and os.path.exists(submission):
        data = json.load(open(submission, encoding="utf-8"))
        reader = re.compile(r"^READER\s+(.+?):", re.M)
        for a in data["answers"]:
            if a["id"] not in qids:
                continue
            blob = "\n".join(str(x) for x in a.get("evidence", []))
            pieces, last, name = [], 0, "unknown"
            for m in reader.finditer(blob):
                pieces.append((name, blob[last:m.start()]))
                name, last = m.group(1), m.end()
            pieces.append((name, blob[last:]))
            for s, c in pieces:
                names += [r["entity"] for r in P.parse_records(c, source=s)]
    names = sorted({n.strip() for n in names if n}, key=len, reverse=True)

    claims = find_text_claims(body, names)
    ranked = [c for c in claims if c["claimed_rank"]]
    print(f"\n{'=' * 74}\n{os.path.basename(doc_path)}\n{'=' * 74}")
    print(f"  entity names available from records: {len(names)}")
    print(f"  claims found in the text:            {len(claims)}")
    print(f"  of those, RANK claims (ordinal):     {len(ranked)}")
    if not claims:
        return

    print(f"\n  --- the RANK claims, which the value ledger cannot see at all ---")
    for c in ranked[:14]:
        print(f"     rank {c['claimed_rank']}  {c['entity'][:34]:<34} :: {c['claim'][:60]!r}"
              f"  scope={P.claim_scope(c['claim'])!r}")

    print(f"\n  --- the two the last run needed ---")
    for want in ("largest national park", "second-highest peak", "second highest"):
        hits = [c for c in claims if want in c["claim"].lower()]
        for c in hits[:4]:
            print(f"     {c['entity'][:34]:<34} :: {c['claim'][:56]!r}  "
                  f"scope={P.claim_scope(c['claim'])!r}  rank={c['claimed_rank']}")

    per = collections.Counter(P._entity_key(c["entity"]) for c in claims)
    print(f"\n  claims per entity: median "
          f"{sorted(per.values())[len(per) // 2]}, max {max(per.values())} "
          f"({max(per, key=per.get)!r})")


if __name__ == "__main__":
    probe(os.path.join(HERE, "nationalparks_europe.txt")
          if os.path.exists(os.path.join(HERE, "nationalparks_europe.txt"))
          else "/Users/yoyomon/Desktop/THU_hackathon/nationalparks_europe.txt",
          os.environ.get("SUBMISSION",
                         "/Users/yoyomon/Downloads/submission_g12_parks-2.json"),
          ("d10", "d11", "d12"))
