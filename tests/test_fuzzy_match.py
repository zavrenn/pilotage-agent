import unittest

from pilotage.tools.fuzzy_match import fuzzy_find_and_replace


class FuzzyMutationSafetyTests(unittest.TestCase):
    def test_similarity_only_block_is_never_mutated(self):
        original = (
            "def target():\n"
            "    debit_account()\n"
            "    send_invoice()\n"
            "    return True\n"
        )
        requested = (
            "def target():\n"
            "    credit_account()\n"
            "    archive_invoice()\n"
            "    return True"
        )

        updated, count, strategy, error = fuzzy_find_and_replace(
            original,
            requested,
            "def target():\n    return False",
        )

        self.assertEqual(updated, original)
        self.assertEqual(count, 0)
        self.assertIsNone(strategy)
        self.assertIn("Could not find a match", error)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
