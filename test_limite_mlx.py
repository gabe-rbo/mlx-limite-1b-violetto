# Unit and Parity Tests for MLX Limite 1B Violetto

import unittest
import numpy as np
import torch
import torch.nn.functional as F
import mlx.core as mx

from model import (
    ModelArgs,
    rms_norm as mx_rms_norm,
    LimiteRotary as MXRotary,
    LimiteMuddMixer as MXMudd,
    Model,
)


def pt_rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class TestLimiteMLX(unittest.TestCase):
    def test_rms_norm(self):
        np_x = np.random.randn(2, 4, 128).astype(np.float32)
        pt_out = pt_rms_norm(torch.tensor(np_x)).numpy()
        mx_out = np.array(mx_rms_norm(mx.array(np_x)))
        np.testing.assert_allclose(pt_out, mx_out, rtol=1e-5, atol=1e-5)
        print("✓ RMSNorm test passed")

    def test_rotary_parity(self):
        head_dim = 128
        n_pairs = 32
        base = 1024.0

        # PyTorch reference rotary
        freq_pt = (1.0 / base) ** torch.linspace(0, 1, steps=n_pairs, dtype=torch.float32)
        freq_pt = freq_pt.repeat_interleave(2)
        freq_pt = torch.cat([freq_pt, freq_pt.new_zeros(head_dim - 2 * n_pairs)])

        def pt_rotate(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            theta = positions.to(torch.float32).view(-1, 1) * freq_pt.view(1, -1)
            cos = theta.cos().view(-1, 1, head_dim)
            sin = theta.sin().view(-1, 1, head_dim)
            sin[:, :, 1::2] *= -1
            x_flip = x.view(*x.shape[:-1], x.shape[-1] // 2, 2).flip(-1).view(x.shape)
            return cos * x + sin * x_flip

        # Inputs: shape [B, S, num_heads, head_dim] -> [1, 4, 2, 128]
        np_x = np.random.randn(4, 2, 128).astype(np.float32)
        pt_positions = torch.arange(0, 4)
        pt_out = pt_rotate(torch.tensor(np_x), pt_positions).numpy()

        # MLX rotary
        args = ModelArgs(head_dim=head_dim, rope_n_pairs=n_pairs, rope_base_local=base)
        mx_rotary = MXRotary(args)
        # MLX expects [B, num_heads, S, head_dim]
        # Transpose from [S, num_heads, head_dim] to [1, num_heads, S, head_dim]
        mx_x = mx.array(np_x)[None].transpose(0, 2, 1, 3)
        mx_out = mx_rotary(mx_x, offset=0)
        # Transpose back to [S, num_heads, head_dim]
        mx_out_np = np.array(mx_out.transpose(0, 2, 1, 3)[0])

        np.testing.assert_allclose(pt_out, mx_out_np, rtol=1e-5, atol=1e-5)
        print("✓ Rotary parity test passed")

    def test_mudd_mixer_parity(self):
        args = ModelArgs(
            num_hidden_layers=4,
            mudd_taps=3,
            mudd_inter=8,
            hidden_size=16,
            mudd_mlp=True,
        )
        mixer = MXMudd(args)

        dense1_np = np.random.randn(8, 16).astype(np.float32)
        dense2_np = np.random.randn(4, 3, 8).astype(np.float32)
        bias_np = np.random.randn(4, 3).astype(np.float32)

        mixer.dense1 = mx.array(dense1_np)
        mixer.dense2 = mx.array(dense2_np)
        mixer.bias = mx.array(bias_np)

        # PyTorch reference combine
        def pt_combine(values, x_cur, layer_idx):
            count = len(values)
            inner = F.gelu(F.linear(pt_rms_norm(x_cur), torch.tensor(dense1_np)))
            dense2 = torch.tensor(dense2_np)
            bias = torch.tensor(bias_np)
            weights = F.linear(inner, dense2[layer_idx, :count]) + bias[layer_idx, :count]
            out = weights[..., 0:1] * values[0]
            for idx in range(1, count):
                out = out + weights[..., idx : idx + 1] * values[idx]
            return out

        val0_np = np.random.randn(1, 3, 16).astype(np.float32)
        val1_np = np.random.randn(1, 3, 16).astype(np.float32)
        x_cur_np = np.random.randn(1, 3, 16).astype(np.float32)

        pt_out = pt_combine(
            [torch.tensor(val0_np), torch.tensor(val1_np)],
            torch.tensor(x_cur_np),
            layer_idx=2,
        ).numpy()

        mx_out = np.array(
            mixer.combine(
                [mx.array(val0_np), mx.array(val1_np)],
                mx.array(x_cur_np),
                layer_idx=2,
            )
        )

        np.testing.assert_allclose(pt_out, mx_out, rtol=1e-5, atol=1e-5)
        print("✓ MUDD mixer parity test passed")

    def test_softcap_parity(self):
        a, b, c = 23.0, 5.0, 7.5
        raw_np = np.random.randn(2, 5, 10).astype(np.float32) * 4.0
        pt_out = (a * torch.sigmoid((torch.tensor(raw_np) + b) / c)).numpy()
        mx_out = np.array(a * mx.sigmoid((mx.array(raw_np) + b) / c))
        np.testing.assert_allclose(pt_out, mx_out, rtol=1e-5, atol=1e-5)
        print("✓ Softcap parity test passed")

    def test_full_model_forward_and_cache(self):
        args = ModelArgs(
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            intermediate_size=128,
            vocab_size=256,
            sliding_window=4,
            rope_n_pairs=4,
            global_layers=[1, 3],
            ve_layers=[0, 2],
            ve_dim=16,
            ve_stored_heads=2,
            ve_gate_channels=4,
            mudd_layers=[2],
            mudd_tap_idx={"2": [0, 2]},
            mudd_inter=8,
            attn_gate_channels=16,
        )

        model = Model(args)
        cache = model.make_cache()

        # Step 1: Prefill first 3 tokens
        tokens = mx.array([[12, 45, 78]])
        logits_prefill = model(tokens, cache=cache)
        self.assertEqual(logits_prefill.shape, (1, 3, 256))

        # Step 2: Next single token decode
        next_token = mx.array([[99]])
        logits_step = model(next_token, cache=cache)
        self.assertEqual(logits_step.shape, (1, 1, 256))
        print("✓ Full model forward and cache stepping passed")

    def test_mlx_lm_load_integration(self):
        import tempfile
        import json
        import shutil
        from pathlib import Path
        import mlx_lm

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            args = ModelArgs(
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                intermediate_size=128,
                vocab_size=256,
                sliding_window=4,
                rope_n_pairs=4,
                global_layers=[1],
                ve_layers=[0],
                ve_dim=16,
                ve_stored_heads=2,
                ve_gate_channels=4,
                mudd_layers=[1],
                mudd_tap_idx={"1": [0, 1]},
                mudd_inter=8,
                attn_gate_channels=16,
            )

            # Write model.py into tmpdir
            shutil.copy2(Path(__file__).parent / "model.py", tmp_path / "model.py")

            # Write config.json
            config_dict = {
                "model_type": "limite",
                "model_file": "model.py",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "intermediate_size": 128,
                "vocab_size": 256,
                "sliding_window": 4,
                "rope_n_pairs": 4,
                "global_layers": [1],
                "ve_layers": [0],
                "ve_dim": 16,
                "ve_stored_heads": 2,
                "ve_gate_channels": 4,
                "mudd_layers": [1],
                "mudd_tap_idx": {"1": [0, 1]},
                "mudd_inter": 8,
                "attn_gate_channels": 16,
                "tie_word_embeddings": True,
            }
            with open(tmp_path / "config.json", "w") as f:
                json.dump(config_dict, f)

            # Instantiate model and get flattened weights
            from mlx.utils import tree_flatten
            model = Model(args)
            flat_weights = dict(tree_flatten(model.parameters()))
            mx.save_safetensors(str(tmp_path / "model.safetensors"), flat_weights)

            # Test load_model
            from mlx_lm.utils import load_model
            loaded_model, loaded_config = load_model(tmp_path)
            self.assertEqual(loaded_config["model_type"], "limite")

            # Test forward pass with loaded model
            dummy_in = mx.array([[5, 10, 15]])
            out = loaded_model(dummy_in)
            self.assertEqual(out.shape, (1, 3, 256))
            print("✓ mlx_lm.load_model integration test passed")


if __name__ == "__main__":
    unittest.main()
