# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is derived from PyTorch,
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/_api.py
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Patch for torch.distributed.tensor._api._ToTorchTensor.

Drops the ``ctx.set_materialize_grads(False)`` call introduced in newer PyTorch
that breaks DTensor gradient propagation when a downstream custom autograd
Function returns ``None`` for an input gradient that came from
``DTensor.to_local()``.

Bug chain (reproduced on deepseek_v4 with TP>=2):

1. A custom autograd Function (e.g. the dsv4 sparse lightning indexer wrapper)
   returns ``None`` as the gradient for an input that was produced by
   ``DTensor.to_local()``.
2. ``_ToTorchTensor.backward`` — because forward had called
   ``ctx.set_materialize_grads(False)`` — receives ``None`` and returns
   ``(None, None)``.
3. Autograd then materialises the upstream ``Redistribute`` output gradient as
   a plain ``torch.zeros(...)`` (no DTensor dispatch — autograd materialisation
   does not preserve the DTensor wrapper).
4. ``Redistribute.backward`` accesses ``grad_output._local_tensor`` and crashes
   with ``AttributeError: 'Tensor' object has no attribute '_local_tensor'``.

Removing the ``set_materialize_grads(False)`` call restores the older
PyTorch behaviour: autograd materialises ``None`` to plain-tensor zeros at the
to_local boundary, and the existing ``_ToTorchTensor.backward`` wraps those
zeros back into a DTensor so the upstream ``Redistribute.backward`` sees a
DTensor as expected.
"""

from torch.distributed.tensor import _api as _dt_api


def _patched_to_torch_tensor_forward(ctx, input, grad_placements):
    """Recreate _ToTorchTensor.forward without set_materialize_grads(False)."""
    ctx.dtensor_spec = input._spec
    ctx.grad_placements = grad_placements
    local_tensor = input._local_tensor
    # Fresh view so autograd metadata is not written back into the DTensor's
    # internal _local_tensor (matches upstream behaviour).
    return local_tensor.view_as(local_tensor)


_dt_api._ToTorchTensor.forward = staticmethod(_patched_to_torch_tensor_forward)
