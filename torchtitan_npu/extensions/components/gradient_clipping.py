# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the LICENSE file.

"""Instance-level gradient clipping seam for TorchTitan 0.3.

Based on TorchTitan 086bf6c166ec85c1298eb5596fa9bf95f6a2d840.
The training step follows the fixed upstream implementation, changing only
its clip call to an overridable instance method. Remove after upstream exposes it.
"""

import math
from collections.abc import Iterator

import torch
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.distributed import utils as dist_utils
from torchtitan.observability import structured_logger as sl
from torchtitan.trainer import Trainer


class GradientClippingTrainer(Trainer):
    def clip_grad_norm(self, parameters, max_norm, **kwargs):
        return dist_utils.clip_grad_norm_(parameters, max_norm, **kwargs)

    def train_step(self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]):
        self.optimizers.zero_grad(set_to_none=self.config.training.disable_cuda_graphs)
        # Save per-optimizer-group learning rates for logging
        lr_metrics = self.lr_schedulers.get_metrics()
        should_log = self.metrics_processor.should_log(self.step)

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims
        # All groups form one optimizer step; each group feeds one fwd-bwd call.
        microbatch_groups: list[list[tuple[dict[str, torch.Tensor], torch.Tensor]]] = []
        local_valid_tokens = torch.tensor(0, dtype=torch.int64)
        for _ in range(self.gradient_accumulation_steps):
            microbatches = []
            for _ in range(self.num_pipeline_parallel_microbatches):
                with sl.log_trace_span("fetching_batch"):
                    input_dict, labels = next(data_iterator)
                local_valid_tokens += (labels != IGNORE_INDEX).sum()
                microbatches.append((input_dict, labels))
            microbatch_groups.append(microbatches)
        sl.log_trace_scalar({"local_valid_tokens": int(local_valid_tokens)})

        # Keep the global token count on device so loss normalization does not
        # introduce a CPU synchronization in the training path.
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            global_valid_tokens = dist_utils.dist_sum_tensor(local_valid_tokens.to(self.device), batch_mesh)
        else:
            global_valid_tokens = local_valid_tokens.to(self.device)

        # Process each gradient accumulation step, then free its inputs.
        accumulated_loss: torch.Tensor | None = None
        for microbatches in microbatch_groups:
            input_dict_mbs = []
            label_mbs = []
            for input_dict, labels in microbatches:
                for key, value in input_dict.items():
                    if isinstance(value, torch.Tensor):
                        input_dict[key] = value.to(self.device)
                input_dict_mbs.append(input_dict)
                label_mbs.append(labels.to(self.device))

            if parallel_dims.pp_enabled:
                fwd_bwd_input_dict = input_dict_mbs
                fwd_bwd_labels = label_mbs
            else:
                assert len(input_dict_mbs) == len(label_mbs) == 1
                fwd_bwd_input_dict = input_dict_mbs[0]
                fwd_bwd_labels = label_mbs[0]

            loss = self.forward_backward_step(
                input_dict=fwd_bwd_input_dict,
                labels=fwd_bwd_labels,
                global_valid_tokens=global_valid_tokens,
            )
            if should_log:
                loss = loss.detach()
                if accumulated_loss is None:
                    # Take ownership before the next replay overwrites the
                    # graph-owned output. Later losses accumulate in place.
                    accumulated_loss = loss.clone()
                else:
                    accumulated_loss.add_(loss)

        with sl.log_trace_span("optim"):
            grad_norm = self.clip_grad_norm(
                [p for m in self.model_parts for p in m.parameters()],
                self.config.training.max_norm,
                foreach=True,
                pp_mesh=parallel_dims.get_optional_mesh("pp"),
                ep_enabled=parallel_dims.ep_enabled,
            )
            self.checkpointer.maybe_wait_for_staging()
            self.optimizers.step()
            self.lr_schedulers.step()

        # log metrics
        if not should_log:
            return

        assert accumulated_loss is not None

        with sl.log_trace_span("collect_dist_metrics"):
            sl.log_trace_scalar({"global_valid_tokens": int(global_valid_tokens)})

            if parallel_dims.dp_cp_enabled:
                loss_mesh = parallel_dims.get_optional_mesh("loss")

                # For global_avg_loss, we want the average loss across all ranks:
                # accumulated_loss = local_loss_sum / global_valid_tokens
                # global_avg_loss = sum(local_loss_sum) / global_valid_tokens
                #                 = sum(accumulated_loss)
                #
                # For global_max_loss, we want the max of local average losses across ranks:
                # local_avg_loss = local_loss_sum / local_valid_tokens
                #                = (accumulated_loss * global_valid_tokens) / local_valid_tokens
                # global_max_loss = max(local_avg_loss)
                local_avg_loss = accumulated_loss * global_valid_tokens / local_valid_tokens
                global_avg_loss, global_max_loss, global_ntokens_seen = (
                    dist_utils.dist_sum(accumulated_loss, loss_mesh),
                    dist_utils.dist_max(local_avg_loss, loss_mesh),
                    dist_utils.dist_sum(
                        torch.tensor(self.ntokens_seen, dtype=torch.int64, device=self.device),
                        loss_mesh,
                    ),
                )
            else:
                global_avg_loss = global_max_loss = float(accumulated_loss.item())
                global_ntokens_seen = self.ntokens_seen

        # Crash on invalid loss. global_avg_loss is a SUM reduction, so a infinite
        # loss on any rank propagates here. This reuses the D2H copy already done
        # for logging, so it adds no extra sync.
        # TODO: make this step work even logging is off.
        if not math.isfinite(global_avg_loss):
            raise RuntimeError(
                f"Loss is not finite (global_avg_loss={global_avg_loss}) at step {self.step}. Stopping training."
            )

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            **lr_metrics,
        }
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics=extra_metrics,
        )
