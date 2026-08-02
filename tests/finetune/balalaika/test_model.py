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
import warnings

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from cosyvoice.finetune.balalaika.artifacts import sha256_file
from cosyvoice.llm.llm import CosyVoice3LM, Qwen2Encoder


def _model_api():
    try:
        return import_module("cosyvoice.finetune.balalaika.model")
    except ModuleNotFoundError as exc:
        raise AssertionError("Balalaika model integration module is missing") from exc


def tiny_cosyvoice3_llm(*, mix_ratio: list[int] | None = None) -> CosyVoice3LM:
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
        mix_ratio=mix_ratio or [5, 15],
    )


def _logits(model: CosyVoice3LM) -> torch.Tensor:
    token_ids = torch.tensor([[2, 3, 4]], dtype=torch.long)
    embedded = model.llm.get_input_embeddings()(token_ids)
    hidden, _ = model.llm(embedded, torch.tensor([3]))
    return model.llm_decoder(hidden)


def _saved_adapter_fixture(root: Path):
    api = _model_api()
    base_dir = root / "Fun-CosyVoice3-0.5B-2512"
    adapter_dir = root / "adapter"
    base_dir.mkdir()
    base = tiny_cosyvoice3_llm().eval()
    torch.save(base.state_dict(), base_dir / "llm.pt")
    base._balalaika_base_model_dir = base_dir
    base._balalaika_base_checkpoint_sha256 = sha256_file(base_dir / "llm.pt")
    fresh_base = copy.deepcopy(base)
    adapted = api.inject_lora(base, api.LoraSettings()).eval()
    api.save_adapter(adapted, adapter_dir)
    return api, base_dir, adapter_dir, fresh_base


def _update_adapter_weights_checksum(adapter_dir: Path) -> None:
    manifest_path = adapter_dir / "adapter_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["weights_sha256"] = sha256_file(adapter_dir / "adapter_model.safetensors")
    manifest_path.write_text(json.dumps(manifest))


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
        self.assertEqual(result["target_tokens_per_sample"].tolist(), [2, 1])

    def test_bistream_counts_only_speech_tokens_not_fill_eos_or_padding(self):
        model = tiny_cosyvoice3_llm(mix_ratio=[1, 3])
        model.llm_decoder.weight.data.zero_()
        batch = {
            "text_token": torch.tensor([[2, 0]]),
            "text_token_len": torch.tensor([1]),
            "speech_token": torch.tensor([[0, 1, 2, 0, 5]]),
            "speech_token_len": torch.tensor([4]),
            "instruct_token": torch.tensor([[5, 0]]),
            "instruct_token_len": torch.tensor([1]),
        }

        with mock.patch("cosyvoice.llm.llm.random.random", return_value=0.0):
            result = model(batch, torch.device("cpu"))

        self.assertEqual(result["correct_tokens_per_sample"].tolist(), [2])
        self.assertEqual(result["target_tokens_per_sample"].tolist(), [4])

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

    def test_audit_rejects_frozen_q_proj_lora(self):
        api = _model_api()
        adapted = api.inject_lora(tiny_cosyvoice3_llm(), api.LoraSettings())
        q_proj = adapted.get_submodule("llm.model.model.layers.0.self_attn.q_proj")
        for name, parameter in q_proj.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(False)

        with self.assertRaisesRegex(RuntimeError, "trainable LoRA"):
            api.audit_trainable_parameters(adapted)

    def test_audit_rejects_missing_target_inventory_entry(self):
        api = _model_api()
        adapted = api.inject_lora(tiny_cosyvoice3_llm(), api.LoraSettings())
        adapted._balalaika_target_modules = tuple(
            target for target in adapted._balalaika_target_modules if not target.endswith("q_proj")
        )

        with self.assertRaisesRegex(RuntimeError, "target inventory"):
            api.audit_trainable_parameters(adapted)

    def test_audit_rejects_duplicated_target_inventory_entry(self):
        api = _model_api()
        adapted = api.inject_lora(tiny_cosyvoice3_llm(), api.LoraSettings())
        adapted._balalaika_target_modules += (adapted._balalaika_target_modules[0],)

        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            api.audit_trainable_parameters(adapted)

    def test_audit_rejects_ambiguous_target_inventory_mapping(self):
        api = _model_api()
        adapted = api.inject_lora(tiny_cosyvoice3_llm(), api.LoraSettings())
        adapted._balalaika_target_modules += (
            "llm.model.model.layers.0.self_attn.q_proj.child",
        )

        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            api.audit_trainable_parameters(adapted)

    def test_audit_rejects_forbidden_qwen_lm_head_lora(self):
        from peft import LoraConfig, inject_adapter_in_model

        api = _model_api()
        adapted = api.inject_lora(tiny_cosyvoice3_llm(), api.LoraSettings())
        forbidden = LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            bias="none",
            target_modules=["llm.model.lm_head"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            inject_adapter_in_model(forbidden, adapted, adapter_name="forbidden")

        with self.assertRaisesRegex(RuntimeError, "lm_head"):
            api.audit_trainable_parameters(adapted)

    def test_audit_rejects_orphan_lora_parameter(self):
        api = _model_api()
        adapted = api.inject_lora(tiny_cosyvoice3_llm(), api.LoraSettings())
        adapted.register_parameter("lora_orphan", torch.nn.Parameter(torch.ones(1)))

        with self.assertRaisesRegex(RuntimeError, "approved target"):
            api.audit_trainable_parameters(adapted)

    def test_audit_and_save_reject_dense_parameter_leaks(self):
        api = _model_api()
        adapted = api.inject_lora(tiny_cosyvoice3_llm(), api.LoraSettings())
        q_proj = adapted.get_submodule("llm.model.model.layers.0.self_attn.q_proj")
        q_proj.base_layer.weight.requires_grad_(True)

        with self.assertRaisesRegex(RuntimeError, "dense trainable"):
            api.audit_trainable_parameters(adapted)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "dense trainable"):
                api.save_adapter(adapted, Path(tmp))

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

    def test_merge_rejects_missing_lora_tensor(self):
        from safetensors.torch import load_file, save_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            api, base_dir, adapter_dir, fresh_base = _saved_adapter_fixture(root)
            weights_path = adapter_dir / "adapter_model.safetensors"
            tensors = load_file(str(weights_path))
            tensors.pop(sorted(tensors)[0])
            save_file(tensors, str(weights_path))
            _update_adapter_weights_checksum(adapter_dir)

            with mock.patch.object(api, "load_base_llm", return_value=fresh_base):
                with self.assertRaisesRegex(ValueError, "missing"):
                    api.merge_adapter(base_dir, adapter_dir, root / "merged.pt")

    def test_merge_rejects_unexpected_lora_tensor(self):
        from safetensors.torch import load_file, save_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            api, base_dir, adapter_dir, fresh_base = _saved_adapter_fixture(root)
            weights_path = adapter_dir / "adapter_model.safetensors"
            tensors = load_file(str(weights_path))
            tensors["orphan.lora_A.weight"] = torch.ones(1)
            save_file(tensors, str(weights_path))
            _update_adapter_weights_checksum(adapter_dir)

            with mock.patch.object(api, "load_base_llm", return_value=fresh_base):
                with self.assertRaisesRegex(ValueError, "unexpected"):
                    api.merge_adapter(base_dir, adapter_dir, root / "merged.pt")

    def test_merge_rejects_corrupted_adapter_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            api, base_dir, adapter_dir, fresh_base = _saved_adapter_fixture(root)
            with (adapter_dir / "adapter_model.safetensors").open("ab") as stream:
                stream.write(b"corrupt")

            with mock.patch.object(api, "load_base_llm", return_value=fresh_base):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    api.merge_adapter(base_dir, adapter_dir, root / "merged.pt")

    def test_merge_rejects_unsupported_manifest_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            api, base_dir, adapter_dir, fresh_base = _saved_adapter_fixture(root)
            manifest_path = adapter_dir / "adapter_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["format_version"] = 999
            manifest_path.write_text(json.dumps(manifest))

            with mock.patch.object(api, "load_base_llm", return_value=fresh_base):
                with self.assertRaisesRegex(ValueError, "manifest"):
                    api.merge_adapter(base_dir, adapter_dir, root / "merged.pt")

    def test_merge_rejects_target_inventory_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            api, base_dir, adapter_dir, fresh_base = _saved_adapter_fixture(root)
            manifest_path = adapter_dir / "adapter_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["target_modules"] = list(reversed(manifest["target_modules"]))
            manifest_path.write_text(json.dumps(manifest))

            with mock.patch.object(api, "load_base_llm", return_value=fresh_base):
                with self.assertRaisesRegex(ValueError, "target inventory"):
                    api.merge_adapter(base_dir, adapter_dir, root / "merged.pt")

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

    def test_load_base_llm_rejects_arbitrary_non_rl_root(self):
        api = _model_api()
        try:
            api.load_base_llm(Path("/models/base"))
        except Exception as exc:
            self.assertIsInstance(exc, ValueError)
            self.assertIn("Fun-CosyVoice3-0.5B-2512", str(exc))
        else:
            self.fail("arbitrary non-RL model root was accepted")


if __name__ == "__main__":
    unittest.main()
