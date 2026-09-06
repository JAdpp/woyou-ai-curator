"""Strict-protocol responses for deterministic control-flow fake providers.

This adapter only decorates test doubles. It is never imported by app code and
must never repair a live provider response. The dedicated condition-contract
tests exercise missing/contradicted/unbound real parser inputs without this
helper. Fixture acceptance remains whatever the fake provider explicitly chose.
"""

from copy import deepcopy
from functools import wraps


def strict_audit_fixture(method):
    @wraps(method)
    async def wrapped(self, prompt, payload):
        output = await method(self, prompt, payload)
        contract = payload.get("retrievalContract", {})
        if not contract.get("conditionContractVersion") or not isinstance(output.get("accepted"), list):
            return output
        result = deepcopy(output)
        result["conditionContractVersion"] = contract["conditionContractVersion"]
        by_id = {item["objectId"]: item for item in payload["candidates"]}
        for decision in result["accepted"]:
            candidate = by_id.get(decision.get("objectId"))
            if candidate is None or not candidate["evidence"]:
                continue
            explicit_predicates = decision.get("predicateEvidence")
            legacy = {row["predicateId"]: row for row in explicit_predicates or []}
            checks = []
            for condition in contract["perObjectConditions"]:
                if condition["kind"] == "predicate" and explicit_predicates is not None:
                    # Missing or unknown legacy checks remain so, preserving
                    # the old fixture's deliberate rejection behaviour.
                    original = legacy.get(condition["id"])
                    if original is not None:
                        checks.append({**original, "conditionId": condition["id"]})
                    continue
                source = next((row for row in candidate["evidence"]
                               if row["id"] in decision.get("evidenceIds", [])),
                              candidate["evidence"][0])
                check = {"conditionId": condition["id"], "status": "supported",
                         "evidenceId": source["id"], "supportingQuote": source["text"]}
                if condition["kind"] in {"object_kind", "material"}:
                    check.update(matchedAlternative=condition["alternatives"][0], relation="exact")
                checks.append(check)
            decision["conditionEvidence"] = checks
        return result

    return wrapped
