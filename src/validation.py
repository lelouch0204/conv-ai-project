"""Artifact checks shared by the runner and stdlib-only audit CLI.

Agreement between policies is necessary but does not establish agreement with a
direct greedy reference. Old text-only artifacts cannot establish token equality.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence


def audit_turn_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[tuple, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    duplicates: List[Dict[str, Any]] = []
    accounting_errors: List[Dict[str, Any]] = []
    for row in rows:
        key = (row["load_regime"], row["seed"], row["conversation_id"], row["turn_id"])
        identity = dict(zip(("load", "seed", "conversation_id", "turn_id"), key))
        policy = row["policy_name"]
        if policy in groups[key]:
            duplicates.append({**identity, "policy": policy})
        groups[key][policy] = row
        if row["drafted_tokens"] != row["accepted_tokens"] + row["canceled_tokens"]:
            accounting_errors.append({**identity, "policy": policy})

    mismatches: List[Dict[str, Any]] = []
    warmup_mismatches: List[Dict[str, Any]] = []
    missing_pairs: List[Dict[str, Any]] = []
    checked = 0
    token_checks = 0
    for key, policies in groups.items():
        identity = dict(zip(("load", "seed", "conversation_id", "turn_id"), key))
        baseline = policies.get("no_draft")
        if baseline is None:
            if len(policies) > 1:
                missing_pairs.append(identity)
            continue
        for name, candidate in policies.items():
            if name == "no_draft":
                continue
            checked += 1
            has_tokens = "response_tokens" in baseline and "response_tokens" in candidate
            token_checks += has_tokens
            field = "response_tokens" if has_tokens else "response_text"
            if baseline[field] != candidate[field]:
                mismatches.append({**identity, "policy": name, "comparison": field,
                                   "baseline": baseline[field], "candidate": candidate[field]})
            if baseline.get("is_warmup", False) != candidate.get("is_warmup", False):
                warmup_mismatches.append({**identity, "policy": name})

    # Detect turns missing from one policy within a load/seed group as well.
    expected_policies: Dict[tuple, set] = defaultdict(set)
    for key, policies in groups.items():
        expected_policies[key[:2]].update(policies)
    for key, policies in groups.items():
        missing = expected_policies[key[:2]] - set(policies)
        if missing:
            missing_pairs.append(dict(zip(("load", "seed", "conversation_id", "turn_id"), key),
                                      missing_policies=sorted(missing)))

    return {
        "turn_rows": len(rows),
        "paired_comparisons": checked,
        "token_comparisons": token_checks,
        "text_only_comparisons": checked - token_checks,
        "output_mismatches": mismatches,
        "warmup_mismatches": warmup_mismatches,
        "accounting_errors": accounting_errors,
        "duplicate_turns": duplicates,
        "missing_pairs": missing_pairs,
        "artifact_consistent": not any((mismatches, warmup_mismatches, accounting_errors, duplicates, missing_pairs)),
        "greedy_reference_status": "not_checked",
    }
