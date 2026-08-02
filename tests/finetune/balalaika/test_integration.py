"""CPU-only end-to-end contract for the Balalaika two-phase recipe."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.finetune.balalaika.tiny_workflow import run_tiny_workflow


class BalalaikaRecipeIntegrationTests(unittest.TestCase):
    def test_tiny_workflow_reaches_verified_export_after_safe_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_tiny_workflow(
                Path(directory),
                source_shards=2,
                phase1_rows=8,
                phase2_rows=8,
                reserved_prompts=2,
                benchmark_rows=8,
            )
            self.assertEqual(result.stage, "complete")
            self.assertTrue(result.final_llm.exists())
            self.assertTrue(result.strict_load_verified)
            self.assertEqual(result.validation_indices, tuple(range(41)))
            self.assertEqual(result.phase_epochs, (2, 3))
            self.assertEqual(result.boundaries_per_epoch, 8)
            self.assertTrue(result.pilot_stopped_before_approval)
            self.assertTrue(result.pilot_checksum_approved)
            self.assertTrue(result.memorization_passed)
            self.assertTrue(result.cache_resumed_after_interruption)
            self.assertTrue(result.phase_resumed_after_interruption)
            self.assertEqual(result.wandb_mode, "offline")
            self.assertEqual(result.upload_calls, 0)


if __name__ == "__main__":
    unittest.main()
