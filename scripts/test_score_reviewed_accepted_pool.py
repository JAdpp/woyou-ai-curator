import unittest

from score_reviewed_accepted_pool import score_rows


def row(ids, **kwargs):
    return {"queryId": "q", "acceptedResults": [{"objectId": oid} for oid in ids],
            "answerability": "partially_supported", **kwargs}


def label(relevance=2, verdict="supports"):
    return {"relevance": relevance, "evidenceVerdict": verdict, "reviewOrigin": "delegated_ai"}


class AcceptedPoolScoringTests(unittest.TestCase):
    def test_unknown_is_not_irrelevant_and_short_lists_are_visible(self):
        score = score_rows([row(["a", "b"])], {("q", "a"): label()})
        self.assertEqual(score["judgedRelevancePrecision"], 1)
        self.assertEqual(score["unjudgedOccurrences"], 1)
        self.assertEqual(score["missingSlotsToFive"], 3)
        self.assertEqual(score["shortListQuestions"], 1)
        self.assertEqual(score["candidateGateAndFiveReviewedEvidenceSupported"], 0)

    def test_empty_list_is_not_perfect_precision(self):
        score = score_rows([row([])], {})
        self.assertIsNone(score["judgedEvidenceSupportPrecision"])
        self.assertEqual(score["emptyListQuestions"], 1)
        self.assertEqual(score["missingSlotsToFive"], 5)

    def test_relevance_does_not_mean_evidence_support(self):
        score = score_rows([row(["a"])], {("q", "a"): label(verdict="insufficient")})
        self.assertEqual(score["relevantOccurrences"], 1)
        self.assertEqual(score["evidenceSupportedOccurrences"], 0)

    def test_five_supported_needs_candidate_gate_and_no_api_error(self):
        labels = {("q", oid): label() for oid in "abcde"}
        good = score_rows([row("abcde")], labels)
        bad = score_rows([row("abcde", apiErrors=[{"code": "unavailable"}])], labels)
        self.assertEqual(good["candidateGateAndFiveReviewedEvidenceSupported"], 1)
        self.assertEqual(bad["candidateGateAndFiveReviewedEvidenceSupported"], 0)

    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            score_rows([row(["a", "a"])], {})


if __name__ == "__main__":
    unittest.main()
