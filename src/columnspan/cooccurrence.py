"""Revisable label co-occurrence analysis over a fixed object universe.

A study fixes three things up front and never lets a correction move them:

* the object universe (the finite population of uint32 object ids),
* the declared working set of labels and its pair family (the test family),
* the family-wise correction method and significance level.

Each label carries two bitmaps: the set of objects *eligible to be observed*
for that label and, nested inside it, the set of objects observed *positive*.
Withdrawing an eligibility observation changes the denominator (the pair's
joint eligible set); withdrawing a positive observation only changes the
contingency counts.  Both kinds of correction are supported, together with
their back-filling counterparts.

Corrections arrive as batches with a correction id and an explicit base
version.  Applying a batch only recomputes counts for pairs involving the
affected labels, but the family-wise correction is always rebuilt for the
whole fixed family and the new version is published atomically: a new count
can never be paired with an old correction.  An independent full
recomputation is available through :meth:`LabelStudy.verify_version` and is
the mechanism the test suite uses to cross-check every evaluation version.

All bitmap handling reuses :class:`columnspan.bitmap.Bitmap`; this module adds
no parallel set implementation.
"""
from __future__ import annotations

import copy as _copy
import hashlib
import json
import math
from dataclasses import dataclass

from .bitmap import Bitmap

CORRECTION_KINDS = frozenset((
    "positive_add", "positive_withdraw",
    "eligibility_add", "eligibility_withdraw",
))
CORRECTION_METHODS = frozenset(("holm", "bh"))
REASON_BLANK = "blank_label"
REASON_CONSTANT = "constant_label"
REASON_SPARSE = "insufficient_joint_observations"
NOT_ESTIMABLE_REASONS = frozenset((REASON_BLANK, REASON_CONSTANT, REASON_SPARSE))


class CooccurrenceError(ValueError):
    """Base class for study-level validation errors."""


class StaleBaseline(CooccurrenceError):
    """A correction batch names a base version that is not the current one."""


class DuplicateCorrection(CooccurrenceError):
    """A correction id was already applied under different content."""


class IdentityMismatch(CooccurrenceError):
    """A resume/restore token does not bind to the same study evidence."""


class CorrectionInterrupted(CooccurrenceError):
    """Raised inside a correction transaction to model an interrupted run."""

    def __init__(self, token):
        super().__init__("family correction interrupted; resume required")
        self.token = token


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(obj):
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def log_binomial(n, k):
    """Natural logarithm of C(n, k); -inf outside the valid range."""
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_two_sided(a, b, c, d):
    """Two-sided Fisher exact p-value for a 2x2 table with fixed margins."""
    n = a + b + c + d
    row1, col1 = a + b, a + c
    lo = max(0, row1 - (n - col1))
    hi = min(row1, col1)
    denom = log_binomial(n, row1)

    def log_prob(x):
        return (log_binomial(col1, x)
                + log_binomial(n - col1, row1 - x) - denom)

    log_obs = log_prob(a)
    total = 0.0
    for x in range(lo, hi + 1):
        # Compare in log space with a relative tolerance so that tables tied
        # with the observed probability are included despite float rounding.
        if log_prob(x) <= log_obs + 1e-12:
            total += math.exp(log_prob(x))
    return min(1.0, total)


def family_correction(pvalues, method, alpha):
    """Adjust an ordered list of p-values over one fixed family.

    ``pvalues`` contains ``None`` for pairs that are not estimable; those
    pairs stay in the family denominator (``family_size``) but receive no
    adjusted value.  Returns a list of dictionaries with rank, adjusted
    p-value and significance for each estimable entry.
    """
    if method not in CORRECTION_METHODS:
        raise CooccurrenceError("unknown correction method")
    m = len(pvalues)
    estimable = [(i, p) for i, p in enumerate(pvalues) if p is not None]
    order = sorted(estimable, key=lambda item: item[1])
    adjusted = [None] * m
    if method == "holm":
        running = 0.0
        for rank, (idx, p) in enumerate(order, start=1):
            running = max(running, (m - rank + 1) * p)
            adjusted[idx] = {
                "rank": rank, "p_adj": min(1.0, running),
                "significant": min(1.0, running) < alpha,
            }
    else:
        k = len(order)
        raw = [0.0] * k
        for rank, (idx, p) in enumerate(order, start=1):
            raw[rank - 1] = min(1.0, m * p / rank)
        running = 1.0
        for reverse in range(k - 1, -1, -1):
            running = min(running, raw[reverse])
            idx, p = order[reverse]
            adjusted[idx] = {
                "rank": reverse + 1, "p_adj": running,
                "significant": running < alpha,
            }
    return adjusted


@dataclass(frozen=True)
class _LabelState:
    eligible: Bitmap
    positive: Bitmap


class LabelStudy:
    """Fixed-population, versioned, revisable label co-occurrence study."""

    def __init__(self, universe, labels, method="holm", alpha=0.05,
                 min_joint_observations=2):
        if not isinstance(universe, (list, tuple, set, Bitmap)):
            raise CooccurrenceError("universe must be a collection of uint32 ids")
        self._universe = Bitmap(universe)
        self._universe_ids = frozenset(self._universe.values())
        if not self._universe_ids:
            raise CooccurrenceError("universe must contain at least one object")
        if not isinstance(labels, (list, tuple)) or not labels:
            raise CooccurrenceError("at least one label is required")
        names = [self._validate_name(spec, i) for i, spec in enumerate(labels)]
        if len(set(names)) != len(names):
            raise CooccurrenceError("duplicate label names are not allowed")
        if method not in CORRECTION_METHODS:
            raise CooccurrenceError("unknown correction method")
        if not isinstance(alpha, (int, float)) or not 0.0 < alpha < 1.0:
            raise CooccurrenceError("alpha must lie strictly between 0 and 1")
        if not isinstance(min_joint_observations, int) or min_joint_observations < 1:
            raise CooccurrenceError("min_joint_observations must be positive")
        self._method = method
        self._alpha = float(alpha)
        self._min_n = min_joint_observations
        self._labels = {}
        self._blank = set()
        for name, spec in zip(names, labels):
            state = self._ingest(name, spec)
            self._labels[name] = state
            if not name.strip():
                self._blank.add(name)
        self._pairs = tuple(
            (names[i], names[j])
            for i in range(len(names)) for j in range(i + 1, len(names))
        )
        # Append-only evidence log; entry 0 is the immutable genesis record.
        self._evidence = []
        self._versions = []
        genesis = {
            "kind": "genesis",
            "universe": sorted(self._universe_ids),
            "labels": [
                {"name": name,
                 "eligible": sorted(self._labels[name].eligible.values()),
                 "positive": sorted(self._labels[name].positive.values())}
                for name in names
            ],
            "method": method, "alpha": self._alpha,
            "min_joint_observations": self._min_n,
            "pairs": [list(pair) for pair in self._pairs],
        }
        head = _digest(genesis)
        self._evidence.append({"seq": 0, "record": genesis,
                               "prev_hash": None, "hash": head})
        corrected, _ = self._evaluate_all()
        self._publish(0, None, None, None, head, corrected,
                      affected=(), scope_pairs=self._pairs)

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _validate_name(spec, index):
        if not isinstance(spec, dict):
            raise CooccurrenceError(f"label {index} must be an object")
        name = spec.get("name")
        if not isinstance(name, str):
            raise CooccurrenceError(f"label {index} name must be a string")
        return name

    def _ingest(self, name, spec):
        eligible = self._bitmap_of(spec.get("eligible", ()), name, "eligible")
        positive = self._bitmap_of(spec.get("positive", ()), name, "positive")
        extra = positive.operate(eligible, "difference")
        if any(True for _ in extra.values()):
            raise CooccurrenceError(
                f"label {name!r}: positives must be subset of eligible objects")
        return _LabelState(eligible=eligible, positive=positive)

    def _bitmap_of(self, values, name, field):
        try:
            bmp = Bitmap(values)
        except ValueError as exc:
            raise CooccurrenceError(
                f"label {name!r}: {field} ids must be uint32") from exc
        outside = bmp.operate(self._universe, "difference")
        if any(True for _ in outside.values()):
            raise CooccurrenceError(
                f"label {name!r}: {field} contains objects outside the fixed universe")
        return bmp

    # ------------------------------------------------------------- evaluation
    def _pair_counts(self, left, right):
        ls = self._labels[left]
        rs = self._labels[right]
        joint = ls.eligible.operate(rs.eligible, "intersection")
        n = joint.rank(2 ** 32)
        both = ls.positive.operate(rs.positive, "intersection")
        a = both.rank(2 ** 32)
        left_only = ls.positive.operate(rs.positive, "difference").operate(joint, "intersection")
        right_only = rs.positive.operate(ls.positive, "difference").operate(joint, "intersection")
        b = left_only.rank(2 ** 32)
        c = right_only.rank(2 ** 32)
        return a, b, c, n - a - b - c, n

    def _evaluate_pair(self, left, right):
        result = {
            "pair": [left, right],
            "estimable": False, "reason": None,
            "n_valid": None, "table": None,
            "p_value": None, "odds_ratio": None, "log_odds_ratio": None,
            "phi": None, "direction": None,
            "family_size": len(self._pairs),
            "rank": None, "adjusted_p": None, "significant": False,
        }
        if left in self._blank or right in self._blank:
            result["reason"] = REASON_BLANK
            return result
        a, b, c, d, n = self._pair_counts(left, right)
        result["n_valid"] = n
        result["table"] = {"both_positive": a, "left_only": b,
                           "right_only": c, "both_negative": d}
        if n < self._min_n:
            result["reason"] = REASON_SPARSE
            return result
        if a + b in (0, n) or a + c in (0, n):
            result["reason"] = REASON_CONSTANT
            return result
        result["estimable"] = True
        result["p_value"] = fisher_two_sided(a, b, c, d)
        ha, hb, hc, hd = (x + 0.5 for x in (a, b, c, d))
        odds = (ha * hd) / (hb * hc)
        result["odds_ratio"] = odds
        result["log_odds_ratio"] = math.log(odds)
        denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
        result["phi"] = ((a * d) - (b * c)) / denom if denom else 0.0
        result["direction"] = "positive" if result["phi"] > 0 else (
            "negative" if result["phi"] < 0 else "none")
        return result

    def _apply_correction_to(self, results):
        adjusted = family_correction(
            [r["p_value"] if r["estimable"] else None for r in results],
            self._method, self._alpha)
        for result, adj in zip(results, adjusted):
            if adj is None:
                continue
            result["rank"] = adj["rank"]
            result["adjusted_p"] = adj["p_adj"]
            result["significant"] = adj["significant"]
        return results

    def _evaluate_all(self):
        """Independent full recomputation over the whole fixed family."""
        results = [self._evaluate_pair(left, right) for left, right in self._pairs]
        return self._apply_correction_to(results), results

    def _evaluate_affected(self, affected_labels, prior_results):
        """Recompute only affected pairs, then rebuild the whole correction."""
        affected = frozenset(affected_labels)
        scope = tuple(pair for pair in self._pairs
                      if pair[0] in affected or pair[1] in affected)
        results = []
        for index, pair in enumerate(self._pairs):
            if pair[0] in affected or pair[1] in affected:
                results.append(self._evaluate_pair(*pair))
            else:
                # Deep copy so published historical versions stay immutable.
                results.append(_copy.deepcopy(prior_results[index]))
        return self._apply_correction_to(results), results, scope

    # ------------------------------------------------------------ corrections
    def _validate_batch(self, batch):
        if not isinstance(batch, dict):
            raise CooccurrenceError("correction batch must be an object")
        correction_id = batch.get("correction_id")
        if not isinstance(correction_id, str) or not correction_id:
            raise CooccurrenceError("correction_id is required")
        base_version = batch.get("base_version")
        if not isinstance(base_version, int) or base_version < 0:
            raise CooccurrenceError("base_version must be a non-negative integer")
        changes = batch.get("changes")
        if not isinstance(changes, list) or not changes:
            raise CooccurrenceError("a batch must contain at least one change")
        for i, change in enumerate(changes):
            if not isinstance(change, dict):
                raise CooccurrenceError(f"change {i} must be an object")
            kind = change.get("kind")
            if kind not in CORRECTION_KINDS:
                raise CooccurrenceError(f"change {i}: unknown kind {kind!r}")
            label = change.get("label")
            if label not in self._labels:
                raise CooccurrenceError(
                    f"change {i}: label {label!r} is not in the fixed working set")
            obj = change.get("object")
            if not isinstance(obj, int) or obj not in self._universe_ids:
                raise CooccurrenceError(
                    f"change {i}: object must belong to the fixed universe")
        return correction_id, base_version, changes

    def affected_scope(self, changes):
        """Labels and family pairs touched by a batch, within the working set."""
        if not isinstance(changes, (list, tuple)):
            raise CooccurrenceError("changes must be a list")
        labels = []
        for change in changes:
            if not isinstance(change, dict):
                raise CooccurrenceError("each change must be an object")
            label = change.get("label")
            if label not in self._labels:
                raise CooccurrenceError(
                    f"label {label!r} is not in the fixed working set")
            if label not in labels:
                labels.append(label)
        affected = frozenset(labels)
        pairs = tuple(pair for pair in self._pairs
                      if pair[0] in affected or pair[1] in affected)
        return {"affected_labels": labels, "affected_pairs": [list(p) for p in pairs]}

    def _check_duplicate(self, correction_id, base_version, batch):
        for entry in self._evidence[1:]:
            if entry["record"]["correction_id"] != correction_id:
                continue
            prior = entry["record"]
            if (prior["base_version"] != base_version
                    or prior["changes"] != batch["changes"]):
                raise DuplicateCorrection(
                    f"correction_id {correction_id!r} was already used "
                    "with different content")
            # Identical re-delivery: return the version already published.
            return entry["record"]["version"]
        return None

    def apply_corrections(self, batch, identity_token=None):
        """Apply one idempotent, baseline-checked correction batch.

        Returns the new (or, for an identical re-delivery, existing) version
        number.  ``identity_token`` may bind the call to the study's current
        universe fingerprint and evidence head, as required when resuming.
        """
        with self.correction_transaction(batch, identity_token=identity_token) as tx:
            tx.run()
        return tx.version

    def correction_transaction(self, batch, identity_token=None):
        return _CorrectionTransaction(self, batch, identity_token)

    def _publish(self, version, correction_id, base_version, changes,
                 evidence_hash, corrected, affected, scope_pairs):
        envelope = {
            "version": version,
            "correction_id": correction_id,
            "base_version": base_version,
            "changes": changes,
            "evidence_hash": evidence_hash,
            "universe_fingerprint": self.universe_fingerprint,
            "affected_labels": list(affected),
            "affected_pairs": [list(p) for p in scope_pairs],
            "method": self._method,
            "alpha": self._alpha,
            "family_size": len(self._pairs),
            "estimable_pairs": sum(1 for r in corrected if r["estimable"]),
            "results": _copy.deepcopy(corrected),
        }
        self._versions.append(envelope)
        return envelope

    # ------------------------------------------------------------- inspection
    @property
    def universe_fingerprint(self):
        return _digest({"universe": sorted(self._universe_ids)})

    @property
    def current_version(self):
        return self._versions[-1]["version"]

    @property
    def family(self):
        return {"pairs": [list(p) for p in self._pairs],
                "method": self._method, "alpha": self._alpha,
                "family_size": len(self._pairs)}

    def identity_token(self):
        """Token binding continuation to this population and all evidence."""
        return {
            "universe_fingerprint": self.universe_fingerprint,
            "evidence_head": self._evidence[-1]["hash"],
            "version": self.current_version,
        }

    def result(self, version=None):
        """Immutable snapshot for a version; the old version stays queryable."""
        if version is None:
            version = self.current_version
        if not isinstance(version, int) or not 0 <= version < len(self._versions):
            raise CooccurrenceError("unknown version")
        return _copy.deepcopy(self._versions[version])

    def pair_result(self, left, right, version=None):
        envelope = self.result(version)
        wanted = {(left, right), (right, left)}
        for row in envelope["results"]:
            if tuple(row["pair"]) in wanted:
                return _copy.deepcopy(row)
        raise CooccurrenceError(f"pair {left!r}/{right!r} is not in the fixed family")

    def verify_version(self, version=None):
        """Replay evidence and fully recompute a version from scratch.

        The recomputation deliberately uses no cached counts or corrections,
        so it cannot silently splice new counts onto old adjusted values.
        """
        if version is None:
            version = self.current_version
        if not isinstance(version, int) or not 0 <= version < len(self._versions):
            raise CooccurrenceError("unknown version")
        replayed = LabelStudy._replay(self._evidence[: version + 1])
        expected = replayed.result(version)
        stored = self.result(version)
        if _canonical(expected["results"]) != _canonical(stored["results"]):
            raise CooccurrenceError(
                f"version {version} failed independent full recomputation")
        if expected["evidence_hash"] != stored["evidence_hash"]:
            raise CooccurrenceError(
                f"version {version} evidence hash disagrees with the log")
        if expected["family_size"] != stored["family_size"]:
            raise CooccurrenceError(
                f"version {version} family size changed outside the log")
        return {"version": version, "ok": True,
                "family_size": expected["family_size"],
                "estimable_pairs": expected["estimable_pairs"],
                "evidence_hash": expected["evidence_hash"]}

    @classmethod
    def _replay(cls, evidence):
        """Rebuild a study from its evidence log using full recomputation.

        Every version is evaluated with :meth:`_evaluate_all` over the whole
        family; the scoped publication path is never used here, so verification
        is independent of the optimization that produced the stored results.
        """
        genesis = evidence[0]["record"]
        study = cls(
            universe=genesis["universe"],
            labels=[{"name": spec["name"], "eligible": spec["eligible"],
                     "positive": spec["positive"]} for spec in genesis["labels"]],
            method=genesis["method"], alpha=genesis["alpha"],
            min_joint_observations=genesis["min_joint_observations"])
        # Constructor rebuilds genesis with fresh hashes; adopt the stored chain
        # and replay every correction against an empty version history.
        study._evidence = [_copy.deepcopy(evidence[0])]
        study._versions = []
        head = evidence[0]["hash"]
        corrected, _ = study._evaluate_all()
        study._publish(0, None, None, None, head, corrected,
                       affected=(), scope_pairs=study._pairs)
        for entry in evidence[1:]:
            record = entry["record"]
            affected = sorted({c["label"] for c in record["changes"]})
            study._apply_changes(record["changes"])
            head = entry["hash"]
            corrected, _ = study._evaluate_all()
            scope = tuple(pair for pair in study._pairs
                          if pair[0] in affected or pair[1] in affected)
            study._evidence.append(_copy.deepcopy(entry))
            study._publish(record["version"], record["correction_id"],
                           record["base_version"], record["changes"], head,
                           corrected, affected=affected, scope_pairs=scope)
        return study

    def _apply_changes(self, changes):
        for change in changes:
            kind, label, obj = change["kind"], change["label"], change["object"]
            state = self._labels[label]
            single = Bitmap((obj,))
            if kind == "positive_add":
                if not any(True for _ in state.eligible.operate(
                        single, "intersection").values()):
                    raise CooccurrenceError(
                        f"{kind}: object {obj} is not eligible for label "
                        f"{label!r}")
                self._labels[label] = _LabelState(
                    state.eligible, state.positive.operate(single, "union"))
            elif kind == "positive_withdraw":
                self._labels[label] = _LabelState(
                    state.eligible, state.positive.operate(single, "difference"))
            elif kind == "eligibility_add":
                self._labels[label] = _LabelState(
                    state.eligible.operate(single, "union"), state.positive)
            else:
                self._labels[label] = _LabelState(
                    state.eligible.operate(single, "difference"),
                    state.positive.operate(single, "difference"))

    def diff_versions(self, base_version, new_version=None):
        """Explain every pair-level change between two published versions."""
        new_version = self.current_version if new_version is None else new_version
        old = {tuple(r["pair"]): r for r in self.result(base_version)["results"]}
        new = {tuple(r["pair"]): r for r in self.result(new_version)["results"]}
        changes = []
        for pair in self._pairs:
            before, after = old[pair], new[pair]
            pair_changes = []
            for field in ("n_valid", "table", "p_value", "odds_ratio",
                          "log_odds_ratio", "phi", "rank", "adjusted_p",
                          "significant", "estimable", "reason"):
                if before[field] != after[field]:
                    pair_changes.append({
                        "field": field, "before": before[field],
                        "after": after[field]})
            if pair_changes:
                affected_pairs = {
                    tuple(p) for p in self.result(new_version)["affected_pairs"]}
                directly = pair in affected_pairs
                changes.append({
                    "pair": list(pair), "directly_modified": directly,
                    "changes": pair_changes,
                    "explanation": ("counts changed by correction"
                                    if directly else
                                    "family correction re-ranked over the "
                                    "fixed family; counts unchanged")})
        return {"base_version": base_version, "new_version": new_version,
                "pair_changes": changes}

    # -------------------------------------------------------- (de)serializing
    def to_dict(self):
        return {
            "universe": sorted(self._universe_ids),
            "labels": [{"name": name,
                        "eligible": sorted(state.eligible.values()),
                        "positive": sorted(state.positive.values())}
                       for name, state in self._labels.items()],
            "method": self._method, "alpha": self._alpha,
            "min_joint_observations": self._min_n,
            "evidence": _copy.deepcopy(self._evidence),
            "versions": _copy.deepcopy(self._versions),
        }

    @classmethod
    def from_dict(cls, payload):
        required = ("universe", "labels", "method", "alpha",
                    "min_joint_observations", "evidence", "versions")
        if not isinstance(payload, dict) or any(k not in payload for k in required):
            raise CooccurrenceError("incomplete study payload")
        study = cls(
            universe=payload["universe"],
            labels=[{"name": spec["name"], "eligible": spec["eligible"],
                     "positive": spec["positive"]} for spec in payload["labels"]],
            method=payload["method"], alpha=payload["alpha"],
            min_joint_observations=payload["min_joint_observations"])
        study._evidence = _copy.deepcopy(payload["evidence"])
        study._versions = _copy.deepcopy(payload["versions"])
        # Re-verify the chain and the population binding on restore.
        head = study._evidence[0]["hash"]
        if head != _digest(study._evidence[0]["record"]):
            raise IdentityMismatch("genesis record has been altered")
        for entry in study._evidence[1:]:
            if entry["prev_hash"] != head:
                raise IdentityMismatch("evidence chain is broken")
            head = _digest({"prev": entry["prev_hash"],
                            "record": entry["record"]})
            if head != entry["hash"]:
                raise IdentityMismatch("evidence record has been altered")
        for index, envelope in enumerate(study._versions):
            if envelope["version"] != index:
                raise IdentityMismatch("version sequence is broken")
            if envelope["universe_fingerprint"] != study.universe_fingerprint:
                raise IdentityMismatch("version binds to a different universe")
        return study


class _CorrectionTransaction:
    """Staged correction with whole-family confirmation and interruption token."""

    def __init__(self, study, batch, identity_token=None):
        self._study = study
        self._batch = batch
        self._identity = identity_token
        self.correction_id, self.base_version, self.changes = (
            study._validate_batch(batch))
        self.version = None
        self._scope = None
        self._committed = False
        self._staged_labels = None
        self._staged_evidence = None
        self._staged_head = None
        self._record = None

    def __enter__(self):
        study = self._study
        if self._identity is not None:
            expected = study.identity_token()
            if (self._identity.get("universe_fingerprint")
                    != expected["universe_fingerprint"]
                    or self._identity.get("evidence_head")
                    != expected["evidence_head"]):
                raise IdentityMismatch(
                    "resume token does not bind to the current universe/evidence")
        # An identical re-delivery is idempotent even if it arrives carrying
        # the original base version after the batch was already published.
        duplicate = study._check_duplicate(
            self.correction_id, self.base_version, self._batch)
        if duplicate is not None:
            self.version = duplicate
            self._committed = True
            return self
        if self.base_version != study.current_version:
            raise StaleBaseline(
                f"base_version {self.base_version} is not current "
                f"version {study.current_version}")
        scope = study.affected_scope(self.changes)
        self._scope = scope
        self._staged_labels = _copy.deepcopy(study._labels)
        self._staged_evidence = _copy.deepcopy(study._evidence)
        self._staged_head = study._evidence[-1]["hash"]
        self._record = {
            "correction_id": self.correction_id,
            "base_version": self.base_version,
            "changes": self.changes,
            "scope_pairs": scope["affected_pairs"],
        }
        return self

    def run(self):
        if self._committed:
            return self.version
        study = self._study
        saved_labels = study._labels
        study._labels = self._staged_labels
        try:
            study._apply_changes(self.changes)
            prior = study._versions[-1]["results"]
            results, corrected, scope = study._evaluate_affected(
                self._scope["affected_labels"], prior)
        finally:
            if not self._committed:
                # Keep the staged copy on the transaction until commit; the
                # live study keeps serving its last published version.
                self._staged_labels = study._labels
                study._labels = saved_labels
        record = _copy.deepcopy(self._record)
        record["version"] = study.current_version + 1
        head = _digest({"prev": self._staged_head, "record": record})
        self._staged_evidence.append({
            "seq": len(self._staged_evidence), "record": record,
            "prev_hash": self._staged_head, "hash": head})
        self._staged_results = corrected
        self._staged_scope = scope
        self._staged_head = head
        return record["version"]

    def interrupt(self):
        """Discard staging and return a token for a bound continuation."""
        if self._committed:
            raise CooccurrenceError("batch already committed")
        token = self._study.identity_token()
        token["resume"] = self.correction_id
        raise CorrectionInterrupted(token)

    def commit(self):
        if self._committed:
            return self.version
        if not hasattr(self, "_staged_results"):
            raise CooccurrenceError("transaction must be run before commit")
        study = self._study
        study._labels = self._staged_labels
        study._evidence = self._staged_evidence
        version = study.current_version + 1
        affected = self._scope["affected_labels"]
        study._publish(
            version, self.correction_id, self.base_version, self.changes,
            self._staged_head, self._staged_results,
            affected=affected, scope_pairs=self._staged_scope)
        self.version = version
        self._committed = True
        return version

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None and not self._committed:
            self.commit()
        # On exception: drop staging entirely; the previous version survives.
        return False
