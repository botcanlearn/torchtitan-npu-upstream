# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU unit tests for BaseEPTokenDispatcher.ep_mesh submesh caching (#5323)."""

import copy
import gc
import weakref

import torchtitan.models.common.moe.token_dispatcher as native_token_dispatcher

import torchtitan_npu  # noqa: F401 - verifies package import installs the patch
from torchtitan_npu.patches.torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
)


class _FakeSubMesh:
    """Submesh stub holding a strong ``_root_mesh`` back-reference like PyTorch DeviceMesh."""

    def __init__(self, root_mesh, ep_group_name: str):
        self._root_mesh = root_mesh
        self.ep_group_name = ep_group_name


class _FakeSparseMesh:
    """Minimal sparse DeviceMesh double counting ``mesh["ep"]`` slicing calls."""

    def __init__(self, ep_group_name: str = "ep_source"):
        self._dim_group_names = ["dp_0", ep_group_name]
        self.slice_calls = 0

    def __getitem__(self, axis: str) -> _FakeSubMesh:
        assert axis == "ep"
        self.slice_calls += 1
        return _FakeSubMesh(self, self._dim_group_names[1])


class _SlotsOnlySparseMesh:
    """Read-only sparse mesh double without ``__dict__`` to exercise fallback."""

    __slots__ = ("_dim_group_names", "slice_calls")

    def __init__(self):
        self._dim_group_names = ["dp_0", "ep_slots"]
        self.slice_calls = 0

    def __getitem__(self, axis: str) -> _FakeSubMesh:
        assert axis == "ep"
        self.slice_calls += 1
        return _FakeSubMesh(self, self._dim_group_names[1])


def _make_dispatcher() -> AllToAllTokenDispatcher:
    return AllToAllTokenDispatcher.Config(num_experts=4, top_k=2).build()


def test_ep_mesh_caches_submesh_per_sparse_mesh_instance(monkeypatch):
    dispatcher = _make_dispatcher()
    first_mesh = _FakeSparseMesh("ep_first")
    second_mesh = _FakeSparseMesh("ep_second")
    active_mesh = None

    monkeypatch.setattr(native_token_dispatcher, "spmd_sparse_mesh", lambda: active_mesh)
    assert dispatcher.ep_mesh is None

    active_mesh = first_mesh
    submesh_a1 = dispatcher.ep_mesh
    submesh_a2 = dispatcher.ep_mesh
    assert submesh_a1 is submesh_a2
    assert submesh_a1.ep_group_name == "ep_first"
    assert first_mesh.slice_calls == 1

    active_mesh = second_mesh
    submesh_b1 = dispatcher.ep_mesh
    submesh_b2 = dispatcher.ep_mesh
    assert submesh_b1 is submesh_b2
    assert submesh_b1 is not submesh_a1
    assert submesh_b1.ep_group_name == "ep_second"
    assert second_mesh.slice_calls == 1


def test_ep_mesh_refreshes_cache_after_shallow_copy_group_swap(monkeypatch):
    dispatcher = _make_dispatcher()
    source_mesh = _FakeSparseMesh("ep_source")
    active_mesh = source_mesh

    monkeypatch.setattr(native_token_dispatcher, "spmd_sparse_mesh", lambda: active_mesh)
    source_submesh = dispatcher.ep_mesh
    assert source_submesh.ep_group_name == "ep_source"
    assert source_mesh.slice_calls == 1

    # Match ep_process_group._replace_sparse_ep_mesh shallow-copy + PG swap.
    dedicated_mesh = copy.copy(source_mesh)
    dedicated_mesh._dim_group_names = list(source_mesh._dim_group_names)
    dedicated_mesh._dim_group_names[1] = "ep_dedicated"
    dedicated_mesh.slice_calls = 0

    active_mesh = dedicated_mesh
    dedicated_submesh1 = dispatcher.ep_mesh
    dedicated_submesh2 = dispatcher.ep_mesh
    assert dedicated_submesh1 is dedicated_submesh2
    assert dedicated_submesh1 is not source_submesh
    assert dedicated_submesh1.ep_group_name == "ep_dedicated"
    assert dedicated_mesh.slice_calls == 1

    active_mesh = source_mesh
    assert dispatcher.ep_mesh is source_submesh
    assert source_mesh.slice_calls == 1


def test_ep_mesh_cache_releases_mesh_with_root_backref_on_gc(monkeypatch):
    dispatcher = _make_dispatcher()
    mesh = _FakeSparseMesh("ep_gc")
    mesh_ref = weakref.ref(mesh)

    monkeypatch.setattr(native_token_dispatcher, "spmd_sparse_mesh", lambda: mesh)
    assert dispatcher.ep_mesh.ep_group_name == "ep_gc"

    monkeypatch.setattr(native_token_dispatcher, "spmd_sparse_mesh", lambda: None)
    del mesh
    gc.collect()
    assert mesh_ref() is None


def test_ep_mesh_falls_back_when_mesh_forbids_attribute_assignment(monkeypatch):
    dispatcher = _make_dispatcher()
    slots_mesh = _SlotsOnlySparseMesh()

    monkeypatch.setattr(native_token_dispatcher, "spmd_sparse_mesh", lambda: slots_mesh)
    first = dispatcher.ep_mesh
    second = dispatcher.ep_mesh
    assert first is not second
    assert first.ep_group_name == "ep_slots"
    assert slots_mesh.slice_calls == 2
