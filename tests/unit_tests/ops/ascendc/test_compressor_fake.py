# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise the CANN fake boundary without loading CANN or an NPU kernel."""

import sys
import types

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode, ShapeEnv

from torchtitan_npu.ops.ascendc.compressor import _compressor_forward_fake, load_compressor


@pytest.mark.parametrize("ratio,coff", [(4, 2), (128, 1)])
@pytest.mark.parametrize("shape,batch", [((16, 8), 2), ((3, 8), 3), ((0, 8), 0), ((2, 17, 8), 2)])
def test_compressor_fake_shapes_and_dtypes(ratio, coff, shape, batch):
    with FakeTensorMode():
        x = torch.empty(shape, dtype=torch.bfloat16)
        weight = torch.empty((coff * 16, 8))
        outputs = _compressor_forward_fake(
            x, weight, weight, torch.empty((1, 8, 4 * 16)), torch.empty((ratio, coff * 16)),
            ratio, cu_seqlens=torch.empty((batch + 1,), dtype=torch.int32), coff=coff,
        )
        if len(shape) == 3:
            prefix = (shape[0], (shape[1] + ratio - 1) // ratio)
        else:
            prefix = (min(shape[0], shape[0] // ratio + batch),)
        assert outputs[0].shape == (*prefix, 16)
        assert outputs[0].dtype == torch.bfloat16
        for saved in outputs[1:]:
            assert saved.shape == (*prefix, coff * ratio, 16)
            assert saved.dtype == torch.float32


@pytest.fixture(scope="module")
def cann_boundary():
    @torch.library.custom_op("test_compressor_compat::forward", mutates_args=())
    def forward(x: torch.Tensor, cu: torch.Tensor, ratio: int) -> torch.Tensor:
        rows = min(x.shape[0], x.shape[0] // ratio + cu.shape[0] - 1)
        return x.sum().expand(rows, x.shape[1]).clone()

    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])

    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return torch.ones_like(x) * grad.sum(), None, None

    forward.register_autograd(backward, setup_context=setup_context)
    return forward


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_loader_replaces_unbacked_fake_and_preserves_autograd(monkeypatch, cann_boundary, ratio, checkpointed):
    # Match the failing Python min in CANN's fake, not a traced Python kernel.
    def original_fake(x, cu, ratio):
        return x.new_empty((min(x.shape[0], x.shape[0] // ratio + cu.shape[0] - 1), x.shape[1]))

    cann_boundary.register_fake(original_fake)

    def symbolic_call():
        env = ShapeEnv()
        tokens, documents = env.create_unbacked_symint(), env.create_unbacked_symint()
        with FakeTensorMode(shape_env=env):
            torch._constrain_as_size(tokens, min=1)
            torch._constrain_as_size(documents, min=2)
            return cann_boundary(torch.empty((tokens, 8)), torch.empty((documents,), dtype=torch.int32), ratio)

    with pytest.raises(GuardOnDataDependentSymNode):
        symbolic_call()

    # Adapt the small CPU test schema to CANN's real fake signature. Registration
    # still passes through the production loader and PyTorch dispatcher.
    register_fake = torch.library.register_fake

    def register(op, fake):
        def adapted(x, cu, ratio):
            weight = x.new_empty((x.shape[1], x.shape[1]))
            return fake(x, weight, weight, x, x, ratio, cu_seqlens=cu)[0]
        return register_fake(op, adapted)

    monkeypatch.setattr(torch.library, "register_fake", register)
    namespace = types.SimpleNamespace(_compressor_forward=torch.ops.test_compressor_compat.forward)
    monkeypatch.setattr(torch.ops, "cann_ops_transformer", namespace)
    module = types.ModuleType("cann_ops_transformer.ops.compressor")
    module.compressor = cann_boundary
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert load_compressor() is cann_boundary
    assert load_compressor() is cann_boundary  # Construction for another layer is safe.
    symbolic_call()
    # Reproduce the unsupported sym_min result reported by the A5 fake call.
    # Scope it to fake dispatch: the shape rule must still return valid sizes.
    with monkeypatch.context() as scoped:
        scoped.setattr(torch, "sym_min", lambda *args: NotImplemented)
        symbolic_call()

    from torch.utils.checkpoint import checkpoint

    def run(x, cu, ratio):
        if checkpointed:
            return checkpoint(cann_boundary, x, cu, ratio, use_reentrant=False)
        return cann_boundary(x, cu, ratio)

    compiled = torch.compile(run, backend="aot_eager", fullgraph=True, dynamic=True)
    for tokens, batch in [(16, 2), (8, 8)]:
        x = torch.randn((tokens, 8), requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        cu = torch.zeros(batch + 1, dtype=torch.int32)
        setattr(x, "_dynamo_unbacked_indices", {0})  # noqa: B010
        setattr(cu, "_dynamo_unbacked_indices", {0})  # noqa: B010
        actual = compiled(x, cu, ratio)
        expected = cann_boundary(reference_x, cu, ratio)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(x.grad, reference_x.grad)
