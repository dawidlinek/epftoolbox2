"""The torch core of the neural-network and fine-tuned families (internal; needs the `[torch]` extra).

- `trainer`: `train_members(module_factory, x, y, y_mask, members, ...) -> TrainResult`, the members of one fit trained in a loop
  or a padded vmap stack, each member's numbers its own (see its docstring); `MemberSpec`, `member_split`.
- `optim`: `StackedAdam` (a learning rate per member, an active mask) and the loop's `Adam`, in the research trainer's arithmetic.
- `runtime`: `resolve_device`, `torch_scope(device, tf32=False, threads=1)`, `runtime_fields(...)`, `pin_torch_threads`.
- `finetune`: `fine_tune(model, windows, ...) -> FineTuneResult`, with the cut asserted.
- `groups`: torch stacks, the models of a run that share a stack key trained together in padded stacks on one device, each
  model's records its own run's to the bit (`ModelPipeline.run`; `TORCH_STACKS=off` runs each alone).

Importing this package imports no torch; its modules `trainer` and `finetune` do (`MissingExtraError` without it), and `runtime`
imports it in each function, so `_worker._pool_init` can use it for a torch family only.
"""

_EXPORTS = {
    "train_members": "trainer",
    "MemberSpec": "trainer",
    "TrainResult": "trainer",
    "member_split": "trainer",
    "StackedAdam": "optim",
    "torch_scope": "runtime",
    "runtime_fields": "runtime",
    "resolve_device": "runtime",
    "pin_torch_threads": "runtime",
    "fine_tune": "finetune",
    "FineTuneWindows": "finetune",
    "FineTuneResult": "finetune",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f"{__name__}.{module}"), name)
