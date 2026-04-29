"""Stage 3 RTN precomputation pipeline.

Quantizes every linear layer at all supported bit-widths (2, 3, 4) using
symmetric RTN, then packs the results into formats consumable by the
available kernel backends:

  - **Generic int16** — always produced, used by GEMM fallback.
  - **AutoGPTQ int32** — produced for 2/3/4-bit layers.  The GPTQ-packed
    tensors (qweight + scales + qzeros) are saved alongside generic files
    so that ``assemble.py`` can load whichever format the runtime selects.
  - **Marlin int32** — produced for 4-bit layers only (handled at assembly
    time from the generic pack, see ``assemble.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from prism.profile.inspect import iter_named_linear_layers
from prism.rtn.manifest import save_manifest
from prism.rtn.pack import pack_qweight_for_storage
from prism.rtn.pack_gptq import pack_for_autogptq
from prism.rtn.quantize import quantize_rtn


def precompute_model_rtn(
    model,
    output_dir: Path,
    group_size: int = 128,
    bits: tuple[int, ...] = (2, 3, 4),
) -> dict[str, object]:
    """Quantize all linear layers and save RTN + AutoGPTQ-packed artifacts.

    Returns
    -------
    manifest : dict
        Full manifest describing all produced artifacts, their paths,
        and backend compatibility flags.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "model_id": "mock-transformer",
        "artifact_root": str(output_dir),
        "group_size": group_size,
        "dtype": "int16",
        "supported_bits": list(bits),
        "supported_backends": ["gemm", "autogptq", "marlin"],
        "layers": {},
    }

    for layer_name, module in iter_named_linear_layers(model):
        normalized_name = layer_name.replace(".", "__")
        manifest["layers"][layer_name] = {}

        for bit in bits:
            quantized = quantize_rtn(module.weight.detach(), bits=bit, group_size=group_size)
            layer_dir = output_dir / "layers" / normalized_name / f"{bit}bit"
            layer_dir.mkdir(parents=True, exist_ok=True)

            # --- Generic RTN storage (always) ---
            qweight_path = layer_dir / "qweight.pt"
            scale_path = layer_dir / "scales.pt"
            metadata_path = layer_dir / "metadata.json"

            torch.save(pack_qweight_for_storage(quantized["qweight"]), qweight_path)
            torch.save(quantized["scales"], scale_path)

            metadata = {
                "layer_name": layer_name,
                "bit_width": bit,
                "shape": list(module.weight.shape),
                "group_size": group_size,
                "dtype": "int16",
            }

            # --- AutoGPTQ packing (2/3/4-bit) ---
            autogptq_compatible = False
            try:
                gptq_pack = pack_for_autogptq(
                    qweight=quantized["qweight"],
                    scales=quantized["scales"],
                    bits=bit,
                    group_size=group_size,
                    shape=list(module.weight.shape),
                )
                gptq_qweight_path = layer_dir / "gptq_qweight.pt"
                gptq_scales_path = layer_dir / "gptq_scales.pt"
                gptq_qzeros_path = layer_dir / "gptq_qzeros.pt"

                torch.save(gptq_pack["qweight_gptq"], gptq_qweight_path)
                torch.save(gptq_pack["scales_gptq"], gptq_scales_path)
                torch.save(gptq_pack["qzeros_gptq"], gptq_qzeros_path)

                metadata["gptq_qweight_path"] = str(gptq_qweight_path.relative_to(output_dir))
                metadata["gptq_scales_path"] = str(gptq_scales_path.relative_to(output_dir))
                metadata["gptq_qzeros_path"] = str(gptq_qzeros_path.relative_to(output_dir))
                metadata["pack_factor"] = gptq_pack["pack_factor"]
                autogptq_compatible = True
            except Exception:  # noqa: BLE001
                pass

            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            layer_manifest = {
                "shape": list(module.weight.shape),
                "qweight_path": str(qweight_path.relative_to(output_dir)),
                "scale_path": str(scale_path.relative_to(output_dir)),
                "metadata_path": str(metadata_path.relative_to(output_dir)),
                "marlin_compatible": bit == 4,
                "autogptq_compatible": autogptq_compatible,
            }

            # Add GPTQ artifact paths to manifest for assembly
            if autogptq_compatible:
                layer_manifest["gptq_qweight_path"] = metadata["gptq_qweight_path"]
                layer_manifest["gptq_scales_path"] = metadata["gptq_scales_path"]
                layer_manifest["gptq_qzeros_path"] = metadata["gptq_qzeros_path"]

            manifest["layers"][layer_name][str(bit)] = layer_manifest

    save_manifest(manifest, output_dir)
    return manifest
