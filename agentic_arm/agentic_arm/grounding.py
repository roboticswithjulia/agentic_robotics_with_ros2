#!/usr/bin/env python3
"""
grounding.py -- resolve a phrase to an object, using embeddings.

Replaces the synonym tables. Instead of a hand written list saying that a
"can" is a cylinder, an embedding model turns both phrases into vectors and
we take the nearest one. Nobody has to anticipate the word "coke".

WHY THIS IS A HYBRID

Embeddings are fuzzy, and that cuts both ways. "small red cube" and "large
red cube" differ by one word out of three, so their vectors are nearly
identical and a nearest neighbour search cannot reliably separate them.
Size and rank are therefore still handled exactly, by a short table, and
everything else goes to the embedder:

    exact      size words (small, large), rank words (largest, second)
    embedded   colour, shape, synonyms, brand names, typos, phrasing

That split is the symbol versus embedding tradeoff in miniature. Neither
mechanism wins outright, which is why production systems use both.

STANDALONE USE

Run it directly to check matching quality and tune the threshold before
wiring it into the robot:

    python3 grounding.py                    # test against the default scene
    python3 grounding.py --threshold 0.6    # stricter
    python3 grounding.py --margin 0.01      # allow closer calls
    python3 grounding.py --plain            # bare names, no descriptions
    python3 grounding.py "the blue coke can"
"""

import json
import math
import sys
import urllib.error
import urllib.request

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "nomic-embed-text"

# How close a phrase must be to any object before we accept it at all.
# Below this, nothing in the scene is plausibly what was meant.
DEFAULT_THRESHOLD = 0.55

# How far ahead the best match must be before we call it decisive. Two
# objects within this margin are a genuine ambiguity and get reported as one.
#
# Measured, not guessed. Similarities between short labels compress into a
# narrow band, roughly 0.6 to 0.75, so gaps between a right and wrong answer
# are small. At 0.04 the correct top ranked answer was being rejected for
# phrases like "blue coke can", where the gap to second place was 0.03.
DEFAULT_MARGIN = 0.004

# ---------------------------------------------------------------------------
# The only vocabulary that stays hardcoded, and only because it must be exact.
# ---------------------------------------------------------------------------

SIZE_WORDS = {
    "tiny": "tiny", "mini": "tiny",
    "small": "small", "little": "small",
    "medium": "medium", "mid": "medium",
    "large": "large", "big": "large", "bigger": "large", "long": "large",
}

# Ordered smallest to largest, so a size word the scene does not use can be
# mapped to the nearest one it does. A scene with only small and large cubes
# should still understand "tiny".
SIZE_ORDER = ["tiny", "small", "medium", "large"]

# How much a matching size word shifts a similarity score. Tuned against
# measured values: "big blue block" scores 0.676 for large blue cube and
# 0.715 for blue cuboid, so the bias must exceed 0.04 to let the size word
# win. It must also stay well below the gap that separates different shapes,
# or size would start overriding shape, which is the bug this replaced.
SIZE_BIAS = 0.08

RANK_DIRECTION = {
    "largest": -1, "biggest": -1, "longest": -1, "tallest": -1,
    "widest": -1, "greatest": -1,
    "smallest": 1, "tiniest": 1, "shortest": 1, "littlest": 1,
}

RANK_INDEX = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    "sixth": 6, "6th": 6,
    "seventh": 7, "7th": 7,
    "eighth": 8, "8th": 8,
}

# Anything that looks like an ordinal but is not in the table above. Without
# this check "tenth largest" quietly becomes "largest", because the unknown
# word falls through to the embedder and the missing index defaults to one.
ORDINAL_SUFFIXES = ("th", "st", "nd", "rd")

HOME_WORDS = {"home", "rest", "start", "origin", "neutral"}


def words(text):
    cleaned = "".join(
        ch if ch.isalnum() or ch.isspace() else " " for ch in text.lower())
    return cleaned.split()


def is_home(text):
    """True when the phrase names the rest pose rather than an object."""
    w = [x for x in words(text) if x not in
         {"the", "your", "its", "go", "going", "back", "to", "return",
          "returning", "position", "pose", "and", "then", "please"}]
    return bool(w) and set(w) <= HOME_WORDS


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class EmbeddingError(RuntimeError):
    pass


class Grounder:
    """Resolves free text to one object in a scene."""

    def __init__(self, scene, host=DEFAULT_HOST, model=DEFAULT_MODEL,
                 threshold=DEFAULT_THRESHOLD, margin=DEFAULT_MARGIN,
                 augment=True, log=print):
        self.scene = scene
        self.augment = augment
        self.host = host.rstrip("/")
        self.model = model
        self.threshold = threshold
        self.margin = margin
        self.log = log
        self.vectors = {}
        self._embed_scene()

    # -- embedding transport --------------------------------------------

    def _post(self, path, payload, timeout=30):
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def embed(self, texts):
        """Return one vector per input string.

        Tries the batch endpoint first and falls back to the older single
        prompt endpoint, because which one exists depends on the Ollama
        version a participant happens to have installed.
        """
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)

        try:
            out = self._post("/api/embed",
                             {"model": self.model, "input": batch})
            vectors = out["embeddings"]
        except (urllib.error.HTTPError, KeyError):
            vectors = []
            for text in batch:
                out = self._post("/api/embeddings",
                                 {"model": self.model, "prompt": text})
                vectors.append(out["embedding"])
        except urllib.error.URLError as exc:
            raise EmbeddingError(
                f"cannot reach Ollama at {self.host}: {exc}") from exc

        return vectors[0] if single else vectors

    @staticmethod
    def describe(name, obj):
        """Build a richer phrase to embed than the bare object name.

        "cuboid" is geometry jargon. Nobody says it, so it sits far from the
        everyday words people do use, and "red brick" ends up closer to
        "red cylinder" than to "red cuboid".

        The fix is to embed a description rather than a label. Note what this
        is NOT: it is not the synonym table that used to be here. There is one
        line per SHAPE, four in total, and the rest of the phrase is derived
        from the object's own geometry. Adding a purple pyramid needs one line
        here, not entries across four separate word lists.

        The scalable version of this asks the chat model to write these
        descriptions at startup, which removes the last hand written line.
        """
        shape = obj.get("shape", "")
        color = obj.get("color", "")
        size = obj.get("size")
        dims = obj.get("dims") or (0, 0, 0)

        # One line per shape, in the words people actually use.
        gloss = {
            "cube": "a cube, a square block or box",
            "cuboid": "a rectangular bar, a brick, a slab, an oblong block",
            "cylinder": "a cylinder, a can, a tube, a bottle",
            "sphere": "a sphere, a ball, a round marble",
            "tray": "a flat tray to put things on",
            "shelf": "a raised shelf to put things on",
        }.get(shape, shape)

        # Derived from geometry, not written by hand.
        parts = [name, gloss]
        if dims and max(dims) > 0:
            longest, shortest = max(dims), min(dims)
            if longest > 2.0 * shortest:
                parts.append("elongated, longer than it is wide")
            if size:
                parts.append(f"{size} sized")
            parts.append(f"about {round(longest * 100)} cm across")
        if color:
            parts.append(f"coloured {color}")
        return ", ".join(p for p in parts if p)

    def _embed_scene(self):
        """Embed each object TWICE and keep both vectors.

        Measured on a real model, the two representations fail in opposite
        directions:

          bare name     "red brick" lands on red cylinder. The word cuboid is
                        jargon, so the label sits far from everyday speech.
          description   "blue cylindr" becomes ambiguous. The typo matched a
                        short exact label well, and burying that label in
                        fifteen words of description dilutes it.

        Neither wins outright, so keep both and score against whichever is
        closer. Startup cost doubles, from about 0.3 s to 0.6 s, once. Query
        cost is unchanged: one embedding of the phrase, compared against
        twice as many stored vectors, which is arithmetic on 768 floats.
        """
        names = list(self.scene)
        texts = list(names)
        if self.augment:
            texts += [self.describe(n, self.scene[n]) for n in names]

        try:
            vectors = self.embed(texts)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"embedding model '{self.model}' failed. "
                f"Run: ollama pull {self.model}  ({exc})") from exc

        self.vectors = {n: [v] for n, v in zip(names, vectors[:len(names)])}
        if self.augment:
            for n, v in zip(names, vectors[len(names):]):
                self.vectors[n].append(v)

        per = 2 if self.augment else 1
        self.log(f"embedded {len(names)} objects, {per} vector"
                 f"{'s' if per > 1 else ''} each, with {self.model}")

    def _nearest_size(self, size):
        """Map a requested size onto one the scene actually uses."""
        if size is None:
            return None
        available = {o.get("size") for o in self.scene.values()} - {None}
        if not available or size in available:
            return size if size in available else None
        want = SIZE_ORDER.index(size) if size in SIZE_ORDER else 0
        return min(available,
                   key=lambda a: abs(SIZE_ORDER.index(a) - want)
                   if a in SIZE_ORDER else 99)

    def _size_bias(self, name, size):
        """Nudge, not a gate. Zero for objects that have no size variants."""
        if size is None:
            return 0.0
        own = self.scene[name].get("size")
        if own is None:
            return 0.0
        return SIZE_BIAS if own == size else -SIZE_BIAS

    def set_scene(self, scene):
        """Swap in a new world.

        The scene used to be a constant, so embedding it once at startup was
        enough. Now it arrives on a topic and can change: an object is placed
        somewhere new, or in Phase 2 the camera stops seeing something.

        Re-embedding twenty two names costs about 0.6 s, so only do it when
        the set of NAMES has actually changed. A position moving is common
        and needs no new vectors; an object appearing or disappearing is
        rare and does.
        """
        changed = set(scene) != set(self.scene)
        self.scene = scene
        if changed:
            self._embed_scene()
        return changed

    def score(self, query_vec, name):
        """Best similarity across every representation of this object."""
        return max(cosine(query_vec, v) for v in self.vectors[name])

    # -- resolution ------------------------------------------------------

    def _split_query(self, text):
        """Pull out the parts that must be matched exactly.

        Returns (remainder, size, rank_dir, rank_n). The remainder is what
        gets embedded.
        """
        size = rank_dir = rank_n = None
        rest = []
        for w in words(text):
            if w in RANK_DIRECTION:
                rank_dir = RANK_DIRECTION[w]
            elif w in RANK_INDEX:
                rank_n = RANK_INDEX[w]
            elif w in SIZE_WORDS:
                size = SIZE_WORDS[w]
            else:
                rest.append(w)
        if rank_n and not rank_dir:
            rank_dir = -1

        # An unrecognised ordinal must not be silently discarded.
        if rank_dir and rank_n is None:
            for w in rest:
                if (w.endswith(ORDINAL_SUFFIXES) and len(w) > 3
                        and w not in SIZE_WORDS):
                    rank_n = -1  # sentinel: an ordinal we do not understand
                    break
        # A superlative overrides a size filter: "smallest" is a rank, not a
        # request for objects whose size attribute happens to be "small".
        if rank_dir:
            size = None
        return " ".join(rest), size, rank_dir, rank_n

    def candidates(self, text, limit=6):
        """Ranked plausible matches, best first, above the threshold.

        resolve() answers "which one". This answers "which could it be",
        which is what the caller needs when two independent readings of the
        same phrase disagree.
        """
        if not text or not text.strip():
            return []
        key = text.lower().strip()
        if key in self.scene:
            return [(1.0, key)]

        remainder, size, rank_dir, rank_n = self._split_query(text)
        size = self._nearest_size(size)
        if not remainder.strip():
            return []
        try:
            q = self.embed(remainder)
        except Exception:
            return []
        scored = sorted(((self.score(q, n) + self._size_bias(n, size), n)
                         for n in self.scene), reverse=True)
        return [(sc, n) for sc, n in scored if sc >= self.threshold][:limit]

    def refines(self, text):
        """Scene objects whose name contains every word of a phrase.

        "blue cube" gives small blue cube and large blue cube. It does not
        give large green cube, because green is not blue.

        This is deliberately EXACT, not a similarity search. It is only ever
        applied to the planner's output, which is already written in scene
        vocabulary, so the words are known to be the right ones and the only
        question is which object they leave open.

        Using candidates() here instead was a real bug. That is a cosine
        search over the whole scene, and every cube resembles every other
        cube, so "blue cube" produced a shortlist of six that included the
        green ones. The arm then picked up a green cube while the log said
        blue.
        """
        want = set(words(text)) - set(SIZE_WORDS)
        if not want:
            return []
        return [name for name in self.scene
                if want <= set(words(name))]

    def rank_within(self, text, names):
        """Rank a fixed set of objects by how well a phrase matches them.

        candidates() searches the whole scene and drops anything below the
        threshold. This scores an explicit shortlist instead, which is what
        you need when another part of the system has already narrowed things
        down and you only have to choose between what it left.
        """
        if not text or not text.strip() or not names:
            return []
        key = text.lower().strip()
        if key in names:
            return [(1.0, key)]

        remainder, size, _, _ = self._split_query(text)
        size = self._nearest_size(size)
        if not remainder.strip():
            return []
        try:
            q = self.embed(remainder)
        except Exception:
            return []
        return sorted(((self.score(q, n) + self._size_bias(n, size), n)
                       for n in names if n in self.scene), reverse=True)

    def resolve(self, text):
        """Return (name, reason, note).

        name is set on success, reason explains any failure, note records
        anything worth logging such as the similarity score.
        """
        if not text or not text.strip():
            return None, "empty name", None

        key = text.lower().strip()
        if key in self.scene:
            return key, None, None

        remainder, size, rank_dir, rank_n = self._split_query(text)

        # Size BIASES the score, it does not gate the pool.
        #
        # Hard filtering was tried twice and failed twice. Filtering strictly
        # meant "large red cylinder" left only the four large cubes in the
        # pool, and the arm picked up a cube: the size word silently
        # overrode the shape word. Relaxing the filter to keep objects that
        # have no size then meant "big blue block" lost to blue cuboid,
        # because the size word had no force at all.
        #
        # A bias fixes both. An object of the requested size is nudged up, an
        # object of a different stated size is nudged down, and an object
        # with no size variants is left alone. Shape still decides, and size
        # breaks the tie.
        size = self._nearest_size(size)
        pool = list(self.scene)

        # Embed what is left and score it against the pool.
        if remainder.strip():
            try:
                q = self.embed(remainder)
            except Exception as exc:
                return None, f"embedding failed: {exc}", None
            scored = sorted(
                ((self.score(q, n) + self._size_bias(n, size), n)
                 for n in pool), reverse=True)
        else:
            # Pure rank query such as "the largest". Nothing to embed.
            scored = [(1.0, n) for n in pool]

        if rank_dir:
            return self._by_rank(text, scored, rank_dir, rank_n)

        best_score, best_name = scored[0]
        if best_score < self.threshold:
            return None, (f'nothing in the scene is close to "{text}" '
                          f'(best was {best_name} at {best_score:.2f}, '
                          f'below {self.threshold:.2f})'), None

        if len(scored) > 1 and best_score - scored[1][0] < self.margin:
            return None, (f'"{text}" is ambiguous between {best_name} '
                          f'({best_score:.2f}) and {scored[1][1]} '
                          f'({scored[1][0]:.2f})'), None

        return best_name, None, f"similarity {best_score:.2f}"

    def _by_rank(self, text, scored, direction, n):
        """Order the plausible candidates by volume and index into them."""
        plausible = [name for score, name in scored
                     if score >= self.threshold
                     and self.scene[name].get("graspable", True)]
        if not plausible:
            plausible = [name for score, name in scored
                         if self.scene[name].get("graspable", True)]

        if n == -1:
            return None, (f'"{text}" contains an ordinal I do not know. '
                          f'I understand: {", ".join(sorted(RANK_INDEX))}'), None

        n = n or 1
        if n > len(plausible):
            return None, (f'"{text}" asks for number {n} but only '
                          f'{len(plausible)} object'
                          f'{"s" if len(plausible) != 1 else ""} match'), None

        ordered = sorted(plausible,
                         key=lambda k: self.scene[k].get("volume", 0.0),
                         reverse=(direction == -1))
        chosen = ordered[n - 1]

        vol = self.scene[chosen].get("volume", 0.0)
        tied = [k for k in ordered if self.scene[k].get("volume", 0.0) == vol]
        if len(tied) > 1:
            return None, (f'"{text}" is a tie: {", ".join(sorted(tied))} '
                          f'are the same size'), None

        return chosen, None, f"ranked {n} of {len(plausible)} by size"


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def _demo(shape, color, size, dims, graspable=True):
    return {"shape": shape, "color": color, "size": size, "dims": dims,
            "volume": dims[0] * dims[1] * dims[2], "graspable": graspable}


DEMO_SCENE = {}
for _c in ("red", "blue", "green", "golden"):
    DEMO_SCENE[f"small {_c} cube"] = _demo("cube", _c, "small", (.05, .05, .05))
    DEMO_SCENE[f"large {_c} cube"] = _demo("cube", _c, "large", (.08, .08, .08))
    DEMO_SCENE[f"{_c} cuboid"] = _demo("cuboid", _c, None, (.09, .045, .045))
    DEMO_SCENE[f"{_c} cylinder"] = _demo("cylinder", _c, None, (.05, .05, .11))
    DEMO_SCENE[f"{_c} sphere"] = _demo("sphere", _c, None, (.06, .06, .06))
DEMO_SCENE["tray"] = _demo("tray", "grey", None, (.22, .16, .03), False)
DEMO_SCENE["shelf"] = _demo("shelf", "brown", None, (.20, .16, .03), False)

# (phrase, expected result or None if it should be refused)
CASES = [
    # exact names
    ("blue cylinder", "blue cylinder"),
    # regression: a size word must not eliminate a shape that has no sizes.
    # "large red cylinder" used to resolve to "large red cube".
    ("large red cylinder", "red cylinder"),
    ("bigger golden cylinder", "golden cylinder"),
    ("large green cube", "large green cube"),
    # everyday synonyms, none of which appear anywhere in the code
    ("blue can", "blue cylinder"),
    ("blue coke can", "blue cylinder"),
    ("a blue soda bottle", "blue cylinder"),
    ("blue tube", "blue cylinder"),
    ("the golden ball", "golden sphere"),
    ("gold marble", "golden sphere"),
    ("yellow orb", "golden sphere"),
    ("red brick", "red cuboid"),
    ("green slab", "green cuboid"),
    ("big blue block", "large blue cube"),
    ("tiny green box", "small green cube"),
    ("little red box", "small red cube"),
    # typos
    ("blue cylindr", "blue cylinder"),
    ("goldn sphere", "golden sphere"),
    # destinations
    ("the tray", "tray"),
    ("shelf", "shelf"),
    # ranking
    ("largest cuboid", None),   # all four are identical: a tie is correct
    ("smallest cube", None),
    # must refuse
    ("duck toy", None),
    ("flux capacitor", None),
    ("screwdriver", None),
]


def main():
    args = sys.argv[1:]
    threshold = DEFAULT_THRESHOLD
    augment = True
    if "--plain" in args:
        augment = False
        args.remove("--plain")
    if "--threshold" in args:
        i = args.index("--threshold")
        threshold = float(args[i + 1])
        del args[i:i + 2]
    margin = DEFAULT_MARGIN
    if "--margin" in args:
        i = args.index("--margin")
        margin = float(args[i + 1])
        del args[i:i + 2]

    try:
        g = Grounder(DEMO_SCENE, threshold=threshold,
                     margin=margin, augment=augment)
    except EmbeddingError as exc:
        print(f"\n{exc}\n")
        print(f"Fix:  ollama pull {DEFAULT_MODEL}")
        return 1

    if args:
        name, reason, note = g.resolve(" ".join(args))
        print(f"{name}  ({note})" if name else f"REFUSED: {reason}")
        return 0

    print(f"\nthreshold {g.threshold}   margin {g.margin}   "
          f"augment {g.augment}\n")
    width = max(len(p) for p, _ in CASES)
    fails = 0
    for phrase, want in CASES:
        got, reason, note = g.resolve(phrase)
        ok = got == want
        fails += not ok
        detail = f"{got}  [{note}]" if got else f"REFUSED: {reason}"
        print(f"{'  ' if ok else 'XX'} {phrase:<{width}}  ->  {detail}")
        if not ok:
            # Show the ranking so a threshold or margin problem is visible
            # rather than guessed at.
            rem, size, rd, rn = g._split_query(phrase)
            if rem.strip():
                try:
                    q = g.embed(rem)
                    top = sorted(((g.score(q, n), n)
                                  for n in g.vectors),
                                 reverse=True)[:3]
                    print("      top3: " + "   ".join(
                        f"{n} {sc:.3f}" for sc, n in top))
                except Exception:
                    pass

    print()
    if fails:
        print(f"{fails} of {len(CASES)} did not match expectation.")
        print("Adjust with --threshold; higher refuses more, lower accepts more.")
    else:
        print(f"All {len(CASES)} as expected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
