# CI tooling contracts

CI protocol tests live in this unit-test tree so `.ci/unit_test.sh` collects
these checks with `pytest tests/unit_tests`. They are CPU-only and do not imply
that any NPU training configuration has passed end-to-end.

- `test_ci_request.py`: bounded test-ID/parameter requests, identity-bound artifact.
- `test_ci_entrypoint.py`: registered test resolution and shared runner dispatch.
- `test_ci_recipes.py`: Bash argument expansion for A3 16P and A5 32P/64P.

Hardware training is separately verified by the GitHub Actions runs recorded
in `tests/integration_tests/README.md`.
