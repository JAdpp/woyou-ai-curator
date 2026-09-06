"""Small no-network fixtures for top-five scope and blind review packaging."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from review_recovered_pool_extension import build_pool, GROUPS, Journal, blind_candidate, review_packages


class PoolExtensionTests(unittest.TestCase):
    def setUp(self):
        self.questions = {"q1": {"question": "观察窗帘遮挡了什么？", "category": "open_theme", "requiredCulturalLegs": ["europe"]}}
        self.objects = {f"museum:{n}": {"id": f"museum:{n}", "title": f"object {n}", "culture": "France",
                        "imageUrl": "https://museum.example/image.jpg", "themes": ["derived"],
                        "evidence": [{"id": f"museum:{n}:description", "text": "A curtain hides the bed.",
                                      "reviewed": True, "reviewStatus": "source_exact_match", "supports": "old"}]}
                        for n in range(1,8)}
        self.rows = {group: [{"queryId": "q1", "question": self.questions["q1"]["question"],
                             "acceptedResults": [{"objectId": f"museum:{n}", "score": 90, "rank": n} for n in range(1,7)],
                             "answerability": "supported"}] for group in GROUPS}

    def test_union_top_five_excludes_existing_and_hides_chain(self):
        packages, lineage, summary = build_pool(self.questions, self.objects, {("q1", "museum:1")}, self.rows)
        self.assertEqual(summary["newPairCount"], 4)
        self.assertEqual(summary["sharedNewPairs"], 4)
        payload = packages[0]["payload"]
        self.assertEqual(set(payload), {"question", "requiredCulturalLegs", "candidates"})
        self.assertEqual({candidate["objectId"] for candidate in payload["candidates"]}, {f"museum:{n}" for n in range(2,6)})
        self.assertTrue(packages[0]["requiresImageReview"])
        self.assertTrue(all(len(row["sourceOccurrences"]) == 2 for row in lineage))
        self.assertNotIn("answerability", str(payload))
        self.assertNotIn("originalRank", str(payload))
        self.assertNotIn("hybrid-open", str(payload))

    def test_metadata_excludes_prior_source_review_flags(self):
        value = blind_candidate(self.objects["museum:1"])
        self.assertNotIn("themes", value["metadata"])
        self.assertEqual(value["evidence"], [{"id": "museum:1:description", "text": "A curtain hides the bed."}])

    def test_incomplete_or_missing_accepted_chain_rejected(self):
        broken = {GROUPS[0]: [], GROUPS[1]: self.rows[GROUPS[1]]}
        with self.assertRaises(ValueError):
            build_pool(self.questions, self.objects, set(), broken)
        del self.rows[GROUPS[0]][0]["acceptedResults"]
        with self.assertRaises(ValueError):
            build_pool(self.questions, self.objects, set(), self.rows)

    def test_blind_order_independent_of_chain_rank(self):
        before = build_pool(self.questions, self.objects, set(), self.rows)[0]
        for rows in self.rows.values():
            rows[0]["acceptedResults"] = list(reversed(rows[0]["acceptedResults"][:5]))
        self.assertEqual(before, build_pool(self.questions, self.objects, set(), self.rows)[0])


class FakeProvider:
    model = "fixture-text"
    labels_model = "fixture-vision"

    def __init__(self):
        self.calls = 0

    async def generate_retrieval_audit_json(self, prompt, payload):
        self.calls += 1
        return {"decisions": [{"objectId": candidate["objectId"], "relevance": 3,
                               "evidenceVerdict": "supports", "supportingEvidenceIds": [candidate["evidence"][0]["id"]],
                               "note": "机构文字明确记录窗帘遮挡床铺，可支持有限观察。"}
                              for candidate in payload["candidates"]]}


class ReviewResumeTests(unittest.IsolatedAsyncioTestCase):
    def package(self, requires_image):
        return {"queryId": "q1", "requiresImageReview": requires_image,
                "payload": {"question": "窗帘遮挡了什么？", "requiredCulturalLegs": [],
                            "candidates": [{"objectId": "museum:1", "metadata": {"title": "Room"},
                                            "evidence": [{"id": "museum:1:description", "text": "A curtain hides a bed."}]}]}}

    async def test_resume_reuses_completed_text_review(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            provider = FakeProvider()
            journal = Journal(output, "fixture")
            package = self.package(False)
            await review_packages([package], None, provider, None, journal, 1, False)
            resumed = Journal(output, "fixture")
            await review_packages([package], None, provider, None, resumed, 1, False)
            self.assertEqual(provider.calls, 1)
            final = resumed.latest[("q1", "museum:1", "final")]
            self.assertEqual(final["status"], "judged")
            self.assertFalse(final["imageSeen"])

    async def test_missing_image_is_unjudged_not_text_grade(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory), "fixture")
            dataset = SimpleNamespace(objects={"museum:1": {"id": "museum:1", "imageUrl": "https://museum.example/1.jpg"}})
            failed = AsyncMock(return_value=([], {"museum:1": "image_fetch_failed:fixture"}))
            with patch("review_recovered_pool_extension.review._prepare_vision_images", failed), \
                 patch("review_recovered_pool_extension.asyncio.sleep", AsyncMock()):
                await review_packages([self.package(True)], dataset, FakeProvider(), None, journal, 1, False)
            self.assertEqual(failed.await_count, 2)
            final = journal.latest[("q1", "museum:1", "final")]
            self.assertEqual(final["status"], "unjudged")
            self.assertIsNone(final["decision"])
            self.assertFalse(final["imageSeen"])


if __name__ == "__main__":
    unittest.main()
