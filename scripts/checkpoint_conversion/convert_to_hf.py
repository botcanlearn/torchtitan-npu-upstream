# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import importlib.util
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, HuggingFaceStorageWriter
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.config import TORCH_DTYPE_MAP

from torchtitan_npu.converters.framework.state_dict_update_wrapper import (
    apply_state_dict_update,
)
from torchtitan_npu.converters.kernels.gmm import (
    GMMStateDictUpdater,
    NpuGroupedExpertConverter,
)


def _import_model_module(model_name):
    """Import NPU model registry when present, otherwise fall back upstream."""
    npu_module = f"torchtitan_npu.models.{model_name}"
    if importlib.util.find_spec(npu_module) is not None:
        return importlib.import_module(npu_module)
    return importlib.import_module(f"torchtitan.models.{model_name}")


def _checkpoint_has_gmm(input_dir):
    metadata = FileSystemReader(str(input_dir)).read_metadata()
    return any(".moe.experts.w13" in k for k in metadata.state_dict_metadata)


def _prepare_model_for_checkpoint(model, model_spec, input_dir):
    """Apply GMM conversion/updater for offline DCP checkpoints saved with w13."""
    if not _checkpoint_has_gmm(input_dir):
        return
    NpuGroupedExpertConverter(model_spec).convert(model)
    apply_state_dict_update(GMMStateDictUpdater, model_spec)


def _make_tensors_contiguous(state_dict):
    return {key: value.contiguous() if isinstance(value, torch.Tensor) else value for key, value in state_dict.items()}


@torch.inference_mode()
def convert_to_hf(
    input_dir,
    output_dir,
    model_name,
    model_flavor,
    hf_assets_path,
    export_dtype,
):
    # load model and model args so that we can get the state dict shape
    model_module = _import_model_module(model_name)
    model_spec = model_module.model_registry(model_flavor)
    model_config = model_spec.model

    with torch.device("cpu"):
        model = model_config.build()
    _prepare_model_for_checkpoint(model, model_spec, input_dir)
    model = ModelWrapper(model)

    sd_adapter = model_spec.state_dict_adapter(model_config, hf_assets_path)
    assert sd_adapter is not None, (
        "trying to convert checkpoint from DCP to HF safetensors format, but sd_adapter is not provided."
    )

    # allocate state dict memory with empty weights to load checkpoint
    state_dict = model._get_state_dict()

    dcp.load(
        state_dict,
        checkpoint_id=input_dir,
    )

    # convert state dict tt->hf
    hf_state_dict = sd_adapter.to_hf(state_dict)

    # Keep mapped keys from HF state dict; assign unmapped keys to the first shard.
    fqn_to_index_mapping = getattr(sd_adapter, "fqn_to_index_mapping", None)
    if fqn_to_index_mapping:
        fqn_to_index_mapping = {k: v for k, v in fqn_to_index_mapping.items() if k in hf_state_dict} or None

    if fqn_to_index_mapping:
        default_shard = sorted(set(fqn_to_index_mapping.values()))[0]
        fqn_to_index_mapping.update(
            dict.fromkeys(
                hf_state_dict.keys() - fqn_to_index_mapping.keys(),
                default_shard,
            )
        )

    storage_writer = HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    # map and apply export dtype if needed
    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    if target_dtype != torch.float32:
        hf_state_dict = {k: v.to(target_dtype) for k, v in hf_state_dict.items()}
    hf_state_dict = _make_tensors_contiguous(hf_state_dict)

    dcp.save(
        hf_state_dict,
        storage_writer=storage_writer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCP weights to HF format.")
    parser.add_argument("input_dir", type=Path, help="Input directory with DCP weights.")
    parser.add_argument("output_dir", type=Path, help="Output directory for HF checkpoint.")
    parser.add_argument(
        "--hf_assets_path",
        type=Path,
        help="Path to HF assets directory. This is used to get the model.safetensors.index.json mapping",
        default="./assets/hf/Llama-3.1-8B",
    )
    parser.add_argument("--model_name", type=str, nargs="?", default="llama3")
    parser.add_argument("--model_flavor", type=str, nargs="?", default="8B")
    parser.add_argument(
        "--export_dtype",
        type=str,
        nargs="?",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Export dtype for HF checkpoint (default: float32)",
    )
    args = parser.parse_args()

    convert_to_hf(
        args.input_dir,
        args.output_dir,
        args.model_name,
        args.model_flavor,
        args.hf_assets_path,
        args.export_dtype,
    )
