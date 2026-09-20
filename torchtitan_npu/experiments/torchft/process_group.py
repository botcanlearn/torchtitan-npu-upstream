# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HCCL process-group construction and lifecycle for the TorchFT experiment."""

from datetime import timedelta

import torch
from torch.distributed import ProcessGroup, Store
from torchft.process_group import ProcessGroupWrapper


class ProcessGroupHCCLEx(ProcessGroupWrapper):
    # Inherited from the pinned ProcessGroupWrapper; the dependency is optional
    # in the static-only CodeCheck environment.
    _timeout: timedelta
    _quorum_id: int | None
    _group_rank: int | None
    _global_ranks: list[int] | None

    def __init__(self, timeout=timedelta(seconds=60)):
        super().__init__(timeout)
        self._errored: Exception | None = None

    def errored(self):
        return self._errored

    def getBackendName(self) -> str:  # noqa: N802
        return "torchft-hccl"

    def _create_pg(self, store: Store, rank: int, world_size: int) -> ProcessGroup:
        from torch_npu._C._distributed_c10d import ProcessGroupHCCL  # pyrefly: ignore [missing-import]

        self._errored = None
        options = ProcessGroupHCCL.Options()
        options._timeout = self._timeout
        options.group_id = f"torchft_quorum_{self._quorum_id}_rank_{self._group_rank}"
        if self._global_ranks:
            options.global_ranks_in_group = self._global_ranks
        backend = ProcessGroupHCCL(store, rank, world_size, options)
        backend._set_sequence_number_for_group()
        pg = ProcessGroup(store, rank, world_size)
        pg._set_default_backend(ProcessGroup.BackendType.CUSTOM)
        pg._register_backend(torch.device("npu"), ProcessGroup.BackendType.CUSTOM, backend)
        return pg

    def abort(self, errored: bool = True) -> None:
        if errored:
            self._errored = RuntimeError("aborted")
        pg = self._pg
        if pg is not None:
            self._pg = None
            pg._get_backend(torch.device("npu")).abort()

    def shutdown(self) -> None:
        pg = self._pg
        if pg is not None:
            self._pg = None
            pg._get_backend(torch.device("npu")).shutdown()
