# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch.distributed.pipelining.schedules import (
    get_schedule_class,
    PipelineScheduleSingle,
)
from torchtitan.distributed.pipeline_parallel import (
    build_pipeline_schedule,
    generate_llm_fqn_per_model_part,
    pipeline_module_split,
)
from torchtitan.tools.logging import logger


DEEPSEEK_V4_OUTPUT_MODULES = ("hc_head", "norm", "output")


def _get_trainer_model_name(trainer) -> str:
    config = getattr(trainer, "config", None)
    model_spec = getattr(config, "model_spec", None)
    if model_spec is not None and getattr(model_spec, "name", None):
        return str(model_spec.name)

    job_config = getattr(trainer, "job_config", None)
    model_config = getattr(job_config, "model", None)
    return str(getattr(model_config, "name", ""))


def _is_deepseek_v4_pp_target(trainer) -> bool:
    parallel_dims = getattr(trainer, "parallel_dims", None)
    if not getattr(parallel_dims, "pp_enabled", False):
        return False

    return "deepseek_v4" in _get_trainer_model_name(trainer)


def _with_deepseek_v4_pp_input_ids(trainer, result):
    inputs, labels, extra_inputs, extra_kwargs = result
    extra_inputs = extra_inputs or {}
    extra_kwargs = extra_kwargs or {}

    input_ids = extra_kwargs.get("input_ids")
    if "input_ids" in extra_inputs:
        extra_input_ids = extra_inputs.pop("input_ids")
        if input_ids is None:
            input_ids = extra_input_ids
    if input_ids is None:
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
            raise RuntimeError(
                "DeepSeekV4 PP input_ids injection expects inputs with shape [B, S]."
            )
        input_ids = inputs

    extra_kwargs["input_ids"] = input_ids.detach().long()
    return inputs, labels, extra_inputs, extra_kwargs


def _get_num_virtual_stages(
    parallel_dims,
    parallelism,
    num_layers: int,
    input_weight: int,
    output_weight: int,
) -> int:
    schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)
    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)
    layers_per_stage = parallelism.pipeline_parallel_layers_per_stage

    if layers_per_stage is None:
        stages_per_rank = 1 if is_single_stage_schedule else 2
        return parallel_dims.pp * stages_per_rank

    num_virtual_stages = math.ceil(
        (num_layers + input_weight + output_weight) / layers_per_stage
    )
    model_config_info = (
        f"Model has {num_layers} layers with "
        f"pipeline_parallel_layers_per_stage={layers_per_stage}"
    )
    stage_distribution_info = (
        f"resulting in num_virtual_stages={num_virtual_stages} across "
        f"{parallel_dims.pp} PP ranks"
    )

    if num_virtual_stages % parallel_dims.pp != 0:
        raise ValueError(
            f"Number of virtual stages ({num_virtual_stages}) must be divisible by "
            f"pipeline parallel size ({parallel_dims.pp}). {model_config_info}. "
            "Please adjust pipeline_parallel_layers_per_stage to a value that "
            "results in a number of stages divisible by the PP size."
        )

    stages_per_rank = num_virtual_stages // parallel_dims.pp
    if is_single_stage_schedule and stages_per_rank != 1:
        raise ValueError(
            "Single stage schedule requires exactly 1 stage per rank, but got "
            f"{stages_per_rank} stages per rank. {model_config_info}, "
            f"{stage_distribution_info}."
        )
    if not is_single_stage_schedule and stages_per_rank < 2:
        raise ValueError(
            "Multi-stage schedule requires at least 2 stages per rank, but got "
            f"{stages_per_rank} stages per rank. {model_config_info}, "
            f"{stage_distribution_info}."
        )

    return num_virtual_stages


def generate_deepseek_v4_fqn_per_model_part(
    num_stages: int,
    num_layers: int,
    input_weight: int = 1,
    output_weight: int = 1,
) -> list[list[str]]:
    """Reuse the upstream LLM stage splitter and splice in the DeepSeek-V4 head.

    DeepSeek-V4's only structural difference from a generic decoder LLM is the
    extra MHC head (``hc_head``) that must run on the last stage right before
    ``norm``/``output``.  Upstream's ``generate_llm_fqn_per_model_part`` already
    handles all the layer-budget/weight math identically, so we delegate to it
    and just insert ``hc_head`` into the last stage to keep the full
    ``("hc_head", "norm", "output")`` set together.
    """
    module_names_per_stage = generate_llm_fqn_per_model_part(
        num_stages, num_layers, input_weight, output_weight
    )

    last_stage = module_names_per_stage[-1]
    if "norm" not in last_stage:
        raise ValueError(
            "DeepSeekV4 PP expected upstream splitter to place 'norm' on the last "
            f"stage, but got {last_stage}."
        )
    norm_index = last_stage.index("norm")
    module_names_per_stage[-1] = (
        last_stage[:norm_index] + ["hc_head"] + last_stage[norm_index:]
    )
    return module_names_per_stage


def _validate_deepseek_v4_stage_body(
    stage_idx: int,
    modules: list[str],
    num_virtual_stages: int,
    allowed_modules: set[str],
) -> list[int]:
    if len(modules) != len(set(modules)):
        raise ValueError(
            f"DeepSeekV4 PP stage {stage_idx} has duplicate modules: {modules}"
        )
    if stage_idx not in (0, num_virtual_stages - 1) and not modules:
        raise ValueError(f"DeepSeekV4 PP middle stage {stage_idx} is empty.")

    stage_layers = []
    for module_name in modules:
        if module_name.startswith("layers."):
            stage_layers.append(int(module_name.split(".", 1)[1]))
        elif module_name not in allowed_modules:
            raise ValueError(
                f"DeepSeekV4 PP stage {stage_idx} has unsupported module "
                f"{module_name!r}."
            )

    if stage_layers != sorted(stage_layers):
        raise ValueError(
            f"DeepSeekV4 PP stage {stage_idx} has unordered layers: {modules}"
        )
    return stage_layers


def _validate_deepseek_v4_endpoint_modules(
    module_names_per_stage: list[list[str]],
    output_module_set: set[str],
) -> None:
    for stage_idx, modules in enumerate(module_names_per_stage[:-1]):
        overlap = [name for name in modules if name in output_module_set]
        if overlap:
            raise ValueError(
                "DeepSeekV4 PP output modules must stay on the last stage, but "
                f"stage {stage_idx} has {overlap}: {modules}"
            )

    if "tok_embeddings" not in module_names_per_stage[0]:
        raise ValueError("DeepSeekV4 PP first stage must contain tok_embeddings.")
    for stage_idx, modules in enumerate(module_names_per_stage[1:], start=1):
        if "tok_embeddings" in modules:
            raise ValueError(
                "DeepSeekV4 PP tok_embeddings must stay on the first stage, but "
                f"stage {stage_idx} has it: {modules}"
            )

    last_output_modules = [
        name for name in module_names_per_stage[-1] if name in output_module_set
    ]
    if last_output_modules != list(DEEPSEEK_V4_OUTPUT_MODULES):
        raise ValueError(
            "DeepSeekV4 PP last stage must contain output modules in order "
            f"{list(DEEPSEEK_V4_OUTPUT_MODULES)}, but got {last_output_modules}: "
            f"{module_names_per_stage[-1]}"
        )


def validate_deepseek_v4_stage_modules(
    module_names_per_stage: list[list[str]],
    num_layers: int,
    num_virtual_stages: int,
) -> None:
    if len(module_names_per_stage) != num_virtual_stages:
        raise ValueError(
            f"DeepSeekV4 PP expected {num_virtual_stages} virtual stages, but got "
            f"{len(module_names_per_stage)}: {module_names_per_stage}"
        )

    output_module_set = set(DEEPSEEK_V4_OUTPUT_MODULES)
    allowed_modules = {"tok_embeddings", *output_module_set}
    seen_layers = []

    for stage_idx, modules in enumerate(module_names_per_stage):
        seen_layers.extend(
            _validate_deepseek_v4_stage_body(
                stage_idx,
                modules,
                num_virtual_stages,
                allowed_modules,
            )
        )
    _validate_deepseek_v4_endpoint_modules(module_names_per_stage, output_module_set)

    expected_layers = list(range(num_layers))
    if seen_layers != expected_layers:
        raise ValueError(
            "DeepSeekV4 PP layers must appear exactly once in order. Expected "
            f"{expected_layers}, got {seen_layers} from {module_names_per_stage}"
        )


def _build_deepseek_v4_stage_split(parallel_dims, parallelism, model_config):
    input_weight = parallelism.pipeline_parallel_first_stage_less_layers
    output_weight = parallelism.pipeline_parallel_last_stage_less_layers
    num_virtual_stages = _get_num_virtual_stages(
        parallel_dims,
        parallelism,
        model_config.n_layers,
        input_weight,
        output_weight,
    )

    module_names_per_stage = parallelism.module_fqns_per_model_part
    if module_names_per_stage is None:
        module_names_per_stage = generate_deepseek_v4_fqn_per_model_part(
            num_virtual_stages,
            model_config.n_layers,
            input_weight,
            output_weight,
        )

    validate_deepseek_v4_stage_modules(
        module_names_per_stage,
        model_config.n_layers,
        num_virtual_stages,
    )
    return module_names_per_stage


def _parallelize_deepseek_v4_model_parts(
    stages,
    model_parts,
    *,
    parallel_dims,
    training,
    model_converters,
    parallelism,
    compile_config,
    ac_config,
    dump_folder: str,
    parallelize_fn,
) -> None:
    for i, model_part in enumerate(model_parts):
        model_part = parallelize_fn(
            model_part,
            parallel_dims=parallel_dims,
            training=training,
            model_converters=model_converters,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=ac_config,
            dump_folder=dump_folder,
        )
        model_parts[i] = model_part
        stages[i].submod = model_part


def _split_deepseek_v4_pipeline_model(
    model,
    parallel_dims,
    parallelism,
    device: torch.device,
    module_names_per_stage,
):
    for i, stage_modules in enumerate(module_names_per_stage):
        logger.debug(f"DeepSeekV4 PP stage {i}: {stage_modules}")

    return pipeline_module_split(
        model,
        parallel_dims.get_mesh("pp"),
        parallelism.pipeline_parallel_schedule,
        device,
        module_names_per_stage,
    )


def _build_deepseek_v4_pipeline_schedule(parallelism, training, stages, loss_fn):
    return build_pipeline_schedule(
        parallelism=parallelism,
        local_batch_size=training.local_batch_size,
        stages=stages,
        loss_fn=loss_fn,
    )


def _check_deepseek_v4_pp_model_config(model_config) -> None:
    if getattr(model_config, "num_mtp_modules", 0) > 0:
        raise NotImplementedError(
            "DeepSeekV4 MTP + PP is not supported yet. "
            "Please set training.num_mtp_modules=0 when PP is enabled."
        )


def pipeline_deepseek_v4(
    model,
    *,
    parallel_dims,
    training,
    model_converters,
    parallelism,
    compile_config,
    ac_config,
    dump_folder: str,
    device: torch.device,
    model_config,
    parallelize_fn,
    loss_fn,
):
    # Validate DeepSeek-V4 specific PP limitations before splitting.  In
    # particular, MTP depends on token offset embeddings that are not forwarded
    # through the current PP input_ids side channel.
    _check_deepseek_v4_pp_model_config(model_config)

    # Determine which fully-qualified module names should belong to each
    # virtual pipeline stage.  DeepSeek-V4 keeps tok_embeddings on the first
    # stage and hc_head/norm/output on the last stage, unlike generic LLM
    # splitting.
    module_names_per_stage = _build_deepseek_v4_stage_split(
        parallel_dims,
        parallelism,
        model_config,
    )

    # Split the original model into pipeline stages using the DeepSeek-V4
    # module assignment computed above.
    stages, model_parts = _split_deepseek_v4_pipeline_model(
        model,
        parallel_dims,
        parallelism,
        device,
        module_names_per_stage,
    )

    # Apply TP/FSDP/AC/compile handling to each local model part after the PP
    # split.  Each part only owns a subset of root modules, so its parallelize
    # path must be module-presence aware.
    _parallelize_deepseek_v4_model_parts(
        stages,
        model_parts,
        parallel_dims=parallel_dims,
        training=training,
        model_converters=model_converters,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=ac_config,
        dump_folder=dump_folder,
        parallelize_fn=parallelize_fn,
    )

    # Build the runtime pipeline schedule from the prepared stages and loss
    # function.
    pp_schedule = _build_deepseek_v4_pipeline_schedule(
        parallelism,
        training,
        stages,
        loss_fn,
    )

    # Report whether this rank owns the first or last stage.  The trainer uses
    # these flags to decide which inputs/targets each PP rank should handle.
    has_first_stage = any(stage.is_first for stage in stages)
    has_last_stage = any(stage.is_last for stage in stages)

    return pp_schedule, model_parts, has_first_stage, has_last_stage
