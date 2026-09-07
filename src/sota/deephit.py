"""Single-event DeepHit ported from survival-copula's pycox adapter."""

from __future__ import annotations

import torch


class CauseSpecificNet(torch.nn.Module):
    """Shared trunk with one output network per competing risk."""

    def __init__(self, in_features, num_nodes_shared, num_nodes_indiv,
                 num_risks, out_features, batch_norm=True, dropout=None):
        import torchtuples as tt

        super().__init__()
        self.shared_net = tt.practical.MLPVanilla(
            in_features, num_nodes_shared[:-1], num_nodes_shared[-1],
            batch_norm, dropout,
        )
        self.risk_nets = torch.nn.ModuleList([
            tt.practical.MLPVanilla(
                num_nodes_shared[-1], num_nodes_indiv, out_features,
                batch_norm, dropout,
            )
            for _ in range(num_risks)
        ])

    def forward(self, inputs):
        shared = self.shared_net(inputs)
        return torch.stack([network(shared) for network in self.risk_nets], dim=1)


def make_deephit_single(
    in_features: int, time_bins: int, device, config: dict, label_transform=None
):
    from pycox.models import DeepHitSingle
    import torchtuples as tt

    labtrans = label_transform or DeepHitSingle.label_transform(int(time_bins))
    network = tt.practical.MLPVanilla(
        in_features=in_features,
        num_nodes=list(config.get("num_nodes_shared", [64, 32])),
        out_features=labtrans.out_features,
        batch_norm=bool(config.get("batch_norm", True)),
        dropout=float(config.get("dropout", 0.1)),
    )
    model = DeepHitSingle(
        network,
        tt.optim.Adam,
        device=device,
        alpha=float(config.get("alpha", 0.2)),
        sigma=float(config.get("sigma", 0.1)),
        duration_index=labtrans.cuts,
    )
    model.label_transform = labtrans
    model.optimizer.set_lr(float(config.get("learning_rate", 1e-3)))
    return model


def train_deephit_model(model, x_train, y_train, valid_data, config: dict):
    import torchtuples as tt

    callbacks = []
    if bool(config.get("early_stop", True)):
        callbacks.append(
            tt.callbacks.EarlyStopping(patience=int(config.get("patience", 10)))
        )
    model.fit(
        x_train,
        y_train,
        int(config.get("batch_size", 256)),
        int(config.get("epochs", 200)),
        callbacks,
        bool(config.get("verbose", False)),
        val_data=valid_data,
    )
    return model


__all__ = ["CauseSpecificNet", "make_deephit_single", "train_deephit_model"]
