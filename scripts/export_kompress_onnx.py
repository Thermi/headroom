"""Re-export Kompress ONNX model from PyTorch with GPU-optimized settings.

Usage:
    python scripts/export_kompress_onnx.py                              # default: onnx/kompress-int8.onnx
    python scripts/export_kompress_onnx.py --opset 21                   # newer opset
    python scripts/export_kompress_onnx.py --output /tmp/my-model.onnx  # custom path

    Headroom loads the exported model via HEADROOM_KOMPRESS_ONNX_PATH.
    No HuggingFace upload needed — keep the file local.

Requirements:
    pip install headroom-ai[ml]  (torch, transformers, safetensors)
    pip install onnxruntime onnxruntime.quantization  (for INT8 quantize step)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import torch

# Ensure headroom is importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from headroom.transforms.kompress_compressor import (
    HF_MODEL_ID,
    _get_model_class,
)


def export(
    output_path: str = "onnx/kompress-int8.onnx",
    opset: int = 17,
    device: str = "cpu",
    quantize: bool = True,
) -> None:
    """Export the PyTorch model to ONNX with GPU-friendly optimizations.

    Args:
        output_path: Where to write the .onnx file.
        opset: ONNX opset version (17+ recommended for GPU).
        device: 'cpu' for export (ONNX ops work on any device at runtime).
        quantize: Whether to apply dynamic quantization (INT8) on the
                  exported model via onnxruntime.
    """
    print(f"Loading model {HF_MODEL_ID} ...")
    model_cls = _get_model_class()
    model = model_cls()
    model.eval()

    print(f"Downloading weights for {HF_MODEL_ID} ...")
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    weights_path = hf_hub_download(HF_MODEL_ID, "model.safetensors")
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights from {weights_path}")

    # Export to ONNX with dynamic batch/sequence dimensions so the
    # runtime can accept any input shape without re-exporting.
    batch = 1
    seq = 512  # max sequence length (matches inference max_length)

    dummy_input_ids = torch.randint(0, 1000, (batch, seq), dtype=torch.long)
    dummy_attention_mask = torch.ones((batch, seq), dtype=torch.long)

    # The model returns scores via get_scores().  Trace it manually
    # through the forward pass so ONNX captures both heads.
    class KompressONNX(torch.nn.Module):
        """Wrapper that traces the full inference path for ONNX export."""

        def __init__(self, inner: torch.nn.Module):
            super().__init__()
            self.inner = inner

        def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
            with torch.no_grad():
                hidden = self.inner.encoder(
                    input_ids, attention_mask=attention_mask
                ).last_hidden_state
                # Token head: softmax over 2 classes, take class-1 probability
                token_probs = torch.softmax(self.inner.token_head(hidden), dim=-1)[:, :, 1]
                # Span head: 2D CNN (dummy height dim for CUDA compat)
                span_scores = self.inner.span_conv(hidden.transpose(1, 2).unsqueeze(2))
                span_scores = span_scores.view(hidden.size(0), -1)
                return token_probs * (0.5 + 0.5 * span_scores)

    export_model = KompressONNX(model)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Exporting to ONNX (opset={opset}) ...")
    torch.onnx.export(
        export_model,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["final_scores"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "final_scores": {0: "batch", 1: "seq"},
        },
        opset_version=opset,
        do_constant_folding=True,
        # Embed weights directly in the .onnx file instead of using a
        # separate .onnx.data file.  This makes the model self-contained
        # and avoids file-path issues at load time in Docker.
        external_data=False,
    )
    print(f"Exported to {output_path}")

    # Optional: apply INT8 dynamic quantization via ORT.
    # This produces a smaller, faster model at the cost of minor accuracy loss.
    if quantize:
        try:
            import onnxruntime as ort
            from onnxruntime.quantization import quantize_dynamic, QuantType

            orig_path = output_path
            quant_path = output_path.replace(".onnx", "-int8.onnx")
            if quant_path == orig_path:
                backup = orig_path.replace(".onnx", "-fp32.onnx")
                shutil.move(orig_path, backup)
                orig_path = backup

            quantize_dynamic(
                orig_path,
                quant_path,
                weight_type=QuantType.QInt8,
                per_channel=True,
                reduce_range=False,
            )
            print(f"Quantized INT8 model -> {quant_path}")
            output_path = quant_path
        except ImportError:
            print("onnxruntime.quantization not available; skipping quantize step")
        except Exception as exc:
            print(f"Quantization failed: {exc}; keeping FP32 export")

    # Verify the exported model loads in ORT.
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        inp = {session.get_inputs()[0].name: dummy_input_ids.numpy()}
        inp[session.get_inputs()[1].name] = dummy_attention_mask.numpy()
        out = session.run(None, inp)
        print(f"Verification OK: output shape={out[0].shape}")
    except Exception as exc:
        print(f"Verification failed (ONNX load error): {exc}")
        sys.exit(1)

    print(f"Done — saved to {output_path}")
    print(f"To use this export, set:\n"
          f"  export HEADROOM_KOMPRESS_ONNX_PATH={os.path.abspath(output_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-export Kompress ONNX model with GPU-optimized settings"
    )
    parser.add_argument(
        "--output", default="onnx/kompress-int8.onnx",
        help="Output path for the ONNX file (default: onnx/kompress-int8.onnx)"
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version (default: 17, range: 14-21)"
    )
    parser.add_argument(
        "--no-quantize", action="store_true",
        help="Skip INT8 quantization (keep FP32)"
    )
    args = parser.parse_args()

    export(
        output_path=args.output,
        opset=args.opset,
        quantize=not args.no_quantize,
    )
