"""The two network shapes of plan section 3.11 (D12), as presets of `TorchModel` with the library's `epftoolbox2.nets.MLP`.

- `MLPHourlyModel`: one network per (period, horizon, refit anchor) with one output, on that period's rows. The same as
  `TorchModel(module=MLP, module_params={"hidden": [64]}, output="period", ...)`, forecast for forecast.
- `MLPDailyModel`: one network per (horizon, refit anchor) with an output for every period of the target day (the Lago-style
  DNN shape), on day-level predictors. The same as `TorchModel(module=MLP, module_params={"hidden": [256, 128], "dropout":
  0.1}, output="day", ...)`, forecast for forecast.

Both save their architecture as `hidden`, `dropout` and `activation` (no `module` in the file). The manifest records the
hash of `epftoolbox2/nets.py` and the library's version as `module_hash`, compared on resume. Importing this module imports no
torch.
"""

from typing import Callable, List, Optional, Sequence, Union

from .torch_model import TorchModel

MLP_PATH = "epftoolbox2.nets:MLP"
_ACTIVATIONS = ("relu", "gelu", "tanh", "sigmoid", "softplus", "elu", "selu", "silu", "leaky_relu")  # nets.ACTIVATIONS


class _MLPPreset(TorchModel):
    """What the presets share: `nets.MLP(n_inputs, n_outputs, hidden, dropout, activation)` as the module."""

    _OUTPUT = "day"

    def __init__(
        self,
        predictors: Union[List[Union[str, Callable]], Callable],
        hidden: Sequence[int],
        dropout: float,
        activation: str,
        *,
        name: str,
        module_hash: Optional[str],
        **training,
    ):
        where = f"{type(self).__name__}({name!r})"
        if predictors is None:
            raise ValueError(f"{where}: needs predictors=[...]")
        if isinstance(hidden, (str, bytes)) or not hasattr(hidden, "__iter__"):
            raise TypeError(f"{where}: hidden is a list of layer widths, such as [256, 128], got {hidden!r}")
        widths = [int(width) for width in hidden]
        if any(width < 1 or width != w for width, w in zip(widths, hidden)):
            raise ValueError(f"{where}: hidden needs positive whole widths, got {list(hidden)!r}")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError(f"{where}: dropout must be in [0, 1), got {dropout!r}")
        if activation not in _ACTIVATIONS:
            raise ValueError(f"{where}: activation must be one of {list(_ACTIVATIONS)}, got {activation!r}")
        self._setup(predictors, output=self._OUTPUT, name=name, **training)
        self.hidden = widths
        self.dropout = float(dropout)
        self.activation = activation
        self._saved_module_hash = self._checked_hash(module_hash, where)

    @property
    def output(self) -> str:
        """The class's: "day" for `MLPDailyModel`, "period" for `MLPHourlyModel` (no `output` in a saved file)."""
        return self._OUTPUT

    @property
    def inputs(self) -> None:
        return None

    def _module_spec(self):
        return MLP_PATH, {"hidden": list(self.hidden), "dropout": self.dropout, "activation": self.activation}

    def _module_hash(self) -> str:
        return self._saved_module_hash

    @property
    def _model_kwargs(self):
        return {
            "hidden": list(self.hidden),
            "dropout": self.dropout,
            "activation": self.activation,
            "module_hash": self._module_hash(),
            **self._training_kwargs(),
        }


class MLPHourlyModel(_MLPPreset):
    """One `nets.MLP` per (period, horizon, refit anchor), one output: `TorchModel(module=MLP, output="period", ...)`.

    Args:
        predictors: Full predictor names (per-period columns such as `price_d-1` are fine), a `(horizon)` or `(horizon,
            hour)` function, or a function of `ctx`.
        hidden: Widths of the hidden layers.
        dropout: The rate of each hidden layer's `nets.Dropout` (0: none).
        activation: The hidden layers' activation (`nets.ACTIVATIONS`).
        Every other argument is `TorchModel`'s, with these defaults: a 728-day window refitted weekly ("7D"), 3 seeds.
    """

    _OUTPUT = "period"

    def __init__(
        self,
        predictors: Union[List[Union[str, Callable]], Callable],
        hidden: Sequence[int] = (64,),
        dropout: float = 0.0,
        activation: str = "relu",
        training_window: Union[int, str, List[Union[int, str]]] = 728,
        refit: Optional[str] = "7D",
        seeds: int = 3,
        loss: str = "l1",
        lr: float = 3e-4,
        max_epochs: int = 300,
        patience: int = 20,
        batch_size: int = 64,
        early_stopping: float = 0.2,
        weight_decay: float = 0.0,
        stack: str = "auto",
        pad_to: Optional[int] = None,
        device: str = "auto",
        tf32: bool = False,
        random_state: int = 0,
        newest_target: str = "d-1",
        binary_mask: Optional[str] = None,
        scaler="standard",
        name: str = "mlp_hourly",
        module_hash: Optional[str] = None,
    ):
        super().__init__(
            predictors,
            hidden,
            dropout,
            activation,
            name=name,
            module_hash=module_hash,
            training_window=training_window,
            refit=refit,
            seeds=seeds,
            loss=loss,
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            early_stopping=early_stopping,
            weight_decay=weight_decay,
            stack=stack,
            pad_to=pad_to,
            device=device,
            tf32=tf32,
            random_state=random_state,
            newest_target=newest_target,
            binary_mask=binary_mask,
            scaler=scaler,
        )


class MLPDailyModel(_MLPPreset):
    """One `nets.MLP` per (horizon, refit anchor) with an output per period of the target day (the Lago-style DNN):
    `TorchModel(module=MLP, output="day", ...)`.

    Args:
        predictors: Full names of day-level predictors (one value per local date: `price_h05_d-1` from
            `DailyProfileTransformer`, `is_monday_d+{horizon}`), a `(horizon)` function, or a function of `ctx` that may
            change its list with the horizon and the issue day, not the period.
        hidden: Widths of the hidden layers.
        dropout: The rate of each hidden layer's `nets.Dropout` (0: none).
        activation: The hidden layers' activation (`nets.ACTIVATIONS`).
        Every other argument is `TorchModel`'s, with these defaults: a 728-day window refitted weekly ("7D"), 4 seeds.
    """

    _OUTPUT = "day"

    def __init__(
        self,
        predictors: Union[List[Union[str, Callable]], Callable],
        hidden: Sequence[int] = (256, 128),
        dropout: float = 0.1,
        activation: str = "relu",
        training_window: Union[int, str, List[Union[int, str]]] = 728,
        refit: Optional[str] = "7D",
        seeds: int = 4,
        loss: str = "l1",
        lr: float = 3e-4,
        max_epochs: int = 300,
        patience: int = 20,
        batch_size: int = 64,
        early_stopping: float = 0.2,
        weight_decay: float = 0.0,
        stack: str = "auto",
        pad_to: Optional[int] = None,
        device: str = "auto",
        tf32: bool = False,
        random_state: int = 0,
        newest_target: str = "d-1",
        binary_mask: Optional[str] = None,
        scaler="standard",
        name: str = "mlp_daily",
        module_hash: Optional[str] = None,
    ):
        super().__init__(
            predictors,
            hidden,
            dropout,
            activation,
            name=name,
            module_hash=module_hash,
            training_window=training_window,
            refit=refit,
            seeds=seeds,
            loss=loss,
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            early_stopping=early_stopping,
            weight_decay=weight_decay,
            stack=stack,
            pad_to=pad_to,
            device=device,
            tf32=tf32,
            random_state=random_state,
            newest_target=newest_target,
            binary_mask=binary_mask,
            scaler=scaler,
        )
