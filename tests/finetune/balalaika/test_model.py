"""Tests for the CosyVoice3 Balalaika LoRA integration boundary."""

from __future__ import annotations

import copy
from importlib import import_module
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from cosyvoice.finetune.balalaika.artifacts import sha256_file
from cosyvoice.llm.llm import CosyVoice3LM, Qwen2Encoder


def _model_api():
    try:
        return import_module("cosyvoice.finetune.balalaika.model")
    except ModuleNotFoundError as exc:
        raise AssertionError("Balalaika model integration module is missing") from exc


def tiny_cosyvoice3_llm() -> CosyVoice3LM:
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    encoder = Qwen2Encoder.__new__(Qwen2Encoder)
    torch.nn.Module.__init__(encoder)
    encoder.model = Qwen2ForCausalLM(config)
    return CosyVoice3LM(
        llm_input_size=8,
        llm_output_size=8,
        speech_token_size=6,
        llm=encoder,
        sampling=lambda scores, decoded, sampling: int(scores.argmax()),
        mix_ratio=[5, 15],
    )


def _logits(model: CosyVoice3LM) -> torch.Tensor:
    token_ids = torch.tensor([[2, 3, 4]], dtype=torch.long)
    embedded = model.llm.get_input_embeddings()(token_ids)
    hidden, _ = model.llm(embedded, torch.tensor([3]))
    return model.llm_decoder(hidden)


class ModelIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1986)

    def test_qwen_embedding_accessor_preserves_unadapted_output_and_keys(self):
        model = tiny_cosyvoice3_llm()
        keys_before = tuple(model.state_dict())
        token_ids = torch.tensor([[2, 3, 4]], dtype=torch.long)

        nested = model.llm.model.model.embed_tokens(token_ids)
        stable = model.llm.get_input_embeddings()(token_ids)

        torch.testing.assert_close(stable, nested, rtol=0, atol=0)
        self.assertEqual(tuple(model.state_dict()), keys_before)

    def test_forward_exposes_per_sample_teacher_forced_counts(self):
        model = tiny_cosyvoice3_llm()
        model.llm_decoder.weight.data.zero_()
        batch = {
            "text_token": torch.tensor([[2, 3], [4, 0]]),
            "text_token_len": torch.tensor([2, 1]),
            "speech_token": torch.tensor([[0, 1], [0, 0]]),
            "speech_token_len": torch.tensor([2, 1]),
            "instruct_token": torch.tensor([[5], [6]]),
            "instruct_token_len": torch.tensor([1, 1]),
        }

        with mock.patch("cosyvoice.llm.llm.random.random", return_value=1.0):
            result = model(batch, torch.device("cpu"))

        self.assertEqual(result["correct_tokens_per_sample"].dtype, torch.int64)
        self.assertEqual(result["target_tokens_per_sample"].dtype, torch.int64)
        self.assertEqual(result["correct_tokens_per_sample"].tolist(), [1, 1])
        self.assertEqual(result["target_tokens_per_sample"].tolist(), [3, 2])

    def test_active_modules_receive_lora_and_qwen_head_does_not(self):
        api = _model_api()
        model = tiny_cosyvoice3_llm()

        adapted = api.inject_lora(model, api.LoraSettings())
        audit = api.audit_trainable_parameters(adapted)

        self.assertIs(adapted, model)
        self.assertEqual(
            set(audit.target_modules),
            {
                "llm.model.model.embed_tokens",
                "llm.model.model.layers.0.self_attn.q_proj",
                "llm.model.model.layers.0.self_attn.k_proj",
                "llm.model.model.layers.0.self_attn.v_proj",
                "llm.model.model.layers.0.self_attn.o_proj",
                "llm.model.model.layers.0.mlp.gate_proj",
                "llm.model.model.layers.0.mlp.up_proj",
                "llm.model.model.layers.0.mlp.down_proj",
                "llm_decoder",
                "speech_embedding",
            },
        )
        self.assertIn("llm.model.model.layers.0.self_attn.q_proj", audit.target_modules)
        self.assertIn("llm.model.model.embed_tokens", audit.target_modules)
        self.assertIn("speech_embedding", audit.target_modules)
        self.assertIn("llm_decoder", audit.target_modules)
        self.assertNotIn("llm.model.lm_head", audit.target_modules)
        self.assertEqual(audit.unexpected_dense_parameters, ())
        self.assertTrue(audit.trainable_parameters)
        self.assertTrue(all("lora_" in name for name in audit.trainable_parameters))
        for target in audit.target_modules:
            self.assertTrue(
                any(name.startswith(f"{target}.lora_") for name in audit.trainable_parameters),
                target,
            )
        self.assertFalse(model.llm.model.lm_head.weight.requires_grad)

    def test_lora_initialization_is_an_exact_no_op(self):
        api = _model_api()
        model = tiny_cosyvoice3_llm().eval()
        expected = _logits(model).detach().clone()

        adapted = api.inject_lora(model, api.LoraSettings()).eval()

        torch.testing.assert_close(_logits(adapted), expected, rtol=0, atol=0)

    def test_adapter_save_and_merge_match_active_logits(self):
        api = _model_api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_dir = root / "Fun-CosyVoice3-0.5B-2512"
            adapter_dir = root / "adapter"
            output = root / "merged" / "llm.pt"
            base_dir.mkdir()

            base = tiny_cosyvoice3_llm().eval()
            torch.save(base.state_dict(), base_dir / "llm.pt")
            base._balalaika_base_model_dir = base_dir
            base._balalaika_base_checkpoint_sha256 = sha256_file(base_dir / "llm.pt")
            fresh_base = copy.deepcopy(base)
            adapted = api.inject_lora(base, api.LoraSettings()).eval()
            for name, parameter in adapted.named_parameters():
                if "lora_" in name:
                    parameter.data.fill_(0.05)
            expected = _logits(adapted).detach().clone()

            manifest = api.save_adapter(adapted, adapter_dir)
            with mock.patch.object(api, "load_base_llm", return_value=fresh_base):
                report = api.merge_adapter(base_dir, adapter_dir, output)

            merged = tiny_cosyvoice3_llm().eval()
            merged.load_state_dict(torch.load(output, weights_only=True), strict=True)
            torch.testing.assert_close(_logits(merged), expected, rtol=1e-2, atol=1e-2)
            self.assertEqual(manifest.base_checkpoint_sha256, sha256_file(base_dir / "llm.pt"))
            self.assertEqual(report.output_sha256, sha256_file(output))
            self.assertTrue((adapter_dir / "adapter_model.safetensors").is_file())
            inventory = json.loads((adapter_dir / "adapter_manifest.json").read_text())
            self.assertEqual(inventory["settings"]["r"], 64)
            self.assertNotIn("llm.model.lm_head", inventory["target_modules"])
            self.assertFalse(any("lora_" in key for key in merged.state_dict()))

    def test_load_base_llm_loads_only_cosyvoice3_llm(self):
        api = _model_api()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "Fun-CosyVoice3-0.5B-2512"
            model_dir.mkdir()
            (model_dir / "cosyvoice3.yaml").write_text("test fixture\n")
            expected = tiny_cosyvoice3_llm()
            torch.save(expected.state_dict(), model_dir / "llm.pt")
            fake_hyperpyyaml = types.SimpleNamespace(
                load_hyperpyyaml=lambda stream, overrides: {"llm": tiny_cosyvoice3_llm()}
            )

            with mock.patch.dict(sys.modules, {"hyperpyyaml": fake_hyperpyyaml}):
                loaded = api.load_base_llm(model_dir)

            self.assertIsInstance(loaded, CosyVoice3LM)
            self.assertFalse(loaded.training)
            self.assertEqual(
                loaded._balalaika_base_checkpoint_sha256,
                sha256_file(model_dir / "llm.pt"),
            )

    def test_load_base_llm_rejects_rl_checkpoint(self):
        api = _model_api()
        with self.assertRaisesRegex(ValueError, "base/non-RL"):
            api.load_base_llm(Path("/models/Fun-CosyVoice3-0.5B-2512_RL"))


if __name__ == "__main__":
    unittest.main()
