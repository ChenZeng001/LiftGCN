"""Train and evaluate LiftGCN on independent finite element meshes."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from dataloader import (DATA_ROOT, NODE_FEATURE_DIM, FEMGraphDataset, GraphData,
                        discover_samples, collect_train_stats, preload_dataset)
from models import (StressJoukowskiGCN, JOUKOWSKI_RHO_MAX, JOUKOWSKI_RHO_INIT,
                    JOUKOWSKI_RES_SCALE_INIT)

SEED = 42
REPEAT = 10
TRAIN_RATIO = 0.8
EPOCHS = 50
HIDDEN = 500
GCN_LAYERS = 6
DROPOUT = 0.1
LR = 2e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
SPEC_ORDERS = 10
CACHE_GRAPHS = True
DEVICE = torch.device("cpu")
EPS = 1e-12
OUTPUT_DIR = Path("outputs")

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def graph_spectral_signature(
    y: torch.Tensor,
    lap: torch.Tensor,
    orders: int,
) -> torch.Tensor:
    z, energy = y, []
    for _ in range(orders):
        energy.append(z.square().mean(dim=0))
        z = 0.5 * torch.sparse.mm(lap, z)
    energy = torch.stack(energy, dim=0)
    energy = energy / (energy[0:1] + EPS)
    return torch.log10(energy + EPS)


def graph_spec_nmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    lap: torch.Tensor,
    orders: int,
) -> torch.Tensor:
    pred_spec = graph_spectral_signature(pred, lap, orders)
    target_spec = graph_spectral_signature(target, lap, orders)
    numerator = (pred_spec - target_spec).square().sum(dim=0)
    denominator = target_spec.square().sum(dim=0) + EPS
    return (numerator / denominator).mean()


class MetricAccumulator:
    def __init__(self, channels: int = 1) -> None:
        self.sse = torch.zeros(channels, dtype=torch.float64)
        self.energy = torch.zeros(channels, dtype=torch.float64)
        self.sum_y = torch.zeros(channels, dtype=torch.float64)
        self.sum_y2 = torch.zeros(channels, dtype=torch.float64)
        self.nodes = 0
        self.graphs = 0
        self.spec_sum = 0.0
        self.psnr_sum = {1: 0.0, 5: 0.0, 10: 0.0}
        self.hsg_sum = {1: 0.0, 5: 0.0, 10: 0.0}
        self.hsg_count = {1: 0, 5: 0, 10: 0}

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lap: torch.Tensor,
        coords: torch.Tensor,
        edges: torch.Tensor,
    ) -> None:
        error, target64 = (pred - target).double(), target.double()
        self.sse += error.square().sum(dim=0).cpu()
        self.energy += target64.square().sum(dim=0).cpu()
        self.sum_y += target64.sum(dim=0).cpu()
        self.sum_y2 += target64.square().sum(dim=0).cpu()
        self.nodes += target.shape[0]
        self.graphs += 1
        self.spec_sum += float(
            graph_spec_nmse(pred, target, lap, SPEC_ORDERS).item()
        )
        magnitude = torch.linalg.vector_norm(target, dim=1)
        for k in self.psnr_sum:
            count = max(1, math.ceil(target.shape[0] * k / 100.0))
            index = torch.topk(magnitude, count, largest=True).indices
            mse = (pred[index] - target[index]).square().mean()
            peak = target[index].abs().max()
            self.psnr_sum[k] += float(
                (10.0 * torch.log10((peak.square() + EPS) / (mse + EPS))).item()
            )
            hotspot = torch.zeros(
                target.shape[0], dtype=torch.bool, device=target.device
            )
            hotspot[index] = True
            edge_mask = hotspot[edges[:, 0]] | hotspot[edges[:, 1]]
            selected = edges[edge_mask]
            if selected.numel() == 0:
                continue
            i, j = selected[:, 0], selected[:, 1]
            distance = torch.linalg.vector_norm(coords[i] - coords[j], dim=1).clamp_min(
                EPS
            )
            true_grad = (target[i] - target[j]) / distance[:, None]
            pred_grad = (pred[i] - pred[j]) / distance[:, None]
            hsg_nmse = (
                (pred_grad - true_grad).square().sum()
                / true_grad.square().sum().clamp_min(EPS)
            )
            self.hsg_sum[k] += float(hsg_nmse.item())
            self.hsg_count[k] += 1

    def compute(self) -> dict[str, float]:
        valid_energy = self.energy > 1e-30
        nmse = (self.sse[valid_energy] / self.energy[valid_energy]).mean().item()
        snr = (
            10.0
            * torch.log10(
                self.energy[valid_energy]
                / self.sse[valid_energy].clamp_min(1e-30)
            )
        ).mean().item()
        sst = self.sum_y2 - self.sum_y.square() / max(self.nodes, 1)
        valid_sst = sst > 1e-30
        r2 = (1.0 - self.sse[valid_sst] / sst[valid_sst]).mean().item()
        result = {
            "NMSE": nmse,
            "SNR": snr,
            "R2": r2,
            "SpecNMSE": self.spec_sum / self.graphs,
        }
        for k in self.psnr_sum:
            result[f"PSNR@{k}%"] = self.psnr_sum[k] / self.graphs
            result[f"HSG-NMSE@{k}%"] = self.hsg_sum[k] / max(
                self.hsg_count[k], 1
            )
        return result


def move_graph(
    graph: GraphData,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        graph.x.to(DEVICE, non_blocking=True),
        graph.y.to(DEVICE, non_blocking=True),
        graph.adj.to(DEVICE, non_blocking=True),
        graph.lap.to(DEVICE, non_blocking=True),
    )


def train_one_epoch(
    model: nn.Module,
    dataset: FEMGraphDataset,
    optimizer: torch.optim.Optimizer,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    epoch: int,
    seed: int,
) -> float:
    model.train()
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    total_loss = 0.0
    for index in order:
        x, y, adj, _ = move_graph(dataset[index])
        target = (y - y_mean) / y_std
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, adj)
        loss = F.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: FEMGraphDataset,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    metrics = MetricAccumulator(1)
    for index in range(len(dataset)):
        graph = dataset[index]
        x, y, adj, lap = move_graph(graph)
        coords = graph.coords.to(DEVICE, non_blocking=True)
        edges = graph.edges.to(DEVICE, non_blocking=True)
        target_norm = (y - y_mean) / y_std
        pred_norm = model(x, adj)
        total_loss += F.mse_loss(pred_norm, target_norm).item()
        metrics.update(pred_norm * y_std + y_mean, y, lap, coords, edges)
    return total_loss / len(dataset), metrics.compute()


def clone_state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def split_ids(ids: list[int], seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(ids)[rng.permutation(len(ids))].tolist()
    train_count = min(len(ids) - 1, max(1, int(len(ids) * TRAIN_RATIO)))
    return sorted(shuffled[:train_count]), sorted(shuffled[train_count:])


def run_experiment(
    repeat_index: int,
    seed: int,
    ids: list[int],
    files: dict[str, dict[int, Path]],
) -> tuple[dict[str, float], int]:
    seed_everything(seed)
    train_ids, test_ids = split_ids(ids, seed)
    stats = collect_train_stats(train_ids, files)
    train_set = FEMGraphDataset(train_ids, files, stats, CACHE_GRAPHS)
    test_set = FEMGraphDataset(test_ids, files, stats, CACHE_GRAPHS)
    preload_dataset(train_set, f"Repeat {repeat_index} training set")
    preload_dataset(test_set, f"Repeat {repeat_index} test set")
    model = StressJoukowskiGCN(
        NODE_FEATURE_DIM,
        HIDDEN,
        GCN_LAYERS,
        DROPOUT,
        rho_max=JOUKOWSKI_RHO_MAX,
        rho_init=JOUKOWSKI_RHO_INIT,
        res_scale_init=JOUKOWSKI_RES_SCALE_INIT,
    ).to(DEVICE)

    print(f"device={DEVICE} train={len(train_set)} test={len(test_set)}")
    print(f"trainable_parameters={sum(p.numel() for p in model.parameters()):,}")

    y_mean = torch.from_numpy(stats["stress_mean"]).to(DEVICE)
    y_std = torch.from_numpy(stats["stress_std"]).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    best_epoch, best_test_loss, best_state = 0, float("inf"), None

    for epoch in tqdm(
        range(1, EPOCHS + 1),
        desc=f"Repeat {repeat_index}/{REPEAT} seed={seed}",
        unit="epoch",
    ):
        train_one_epoch(model, train_set, optimizer, y_mean, y_std, epoch, seed)
        scheduler.step()
        test_loss, _ = evaluate(model, test_set, y_mean, y_std)
        if test_loss < best_test_loss:
            best_epoch = epoch
            best_test_loss = test_loss
            best_state = clone_state_to_cpu(model)

    if best_state is None:
        raise RuntimeError(f"Repeat {repeat_index} did not produce a valid model.")

    model.load_state_dict(best_state)
    final_test_loss, final_test_metrics = evaluate(model, test_set, y_mean, y_std)
    torch.save({
        "model_state_dict": best_state,
        "model_config": {"in_dim": NODE_FEATURE_DIM, "hidden": HIDDEN,
                         "layers": GCN_LAYERS, "dropout": DROPOUT},
        "stats": {key: torch.from_numpy(value) for key, value in stats.items()},
        "train_ids": train_ids, "test_ids": test_ids, "seed": seed,
        "best_epoch": best_epoch,
        "metrics": {"loss": final_test_loss, **final_test_metrics},
    }, OUTPUT_DIR / f"repeat_{repeat_index:02d}.pt")
    return {"loss": final_test_loss, **final_test_metrics}, best_epoch


def format_summary_value(name: str, mean: float, std: float) -> str:
    if name == "R2":
        return f"{mean:.5f} ± {std:.5f}"
    if name == "SNR" or name.startswith("PSNR"):
        return f"{mean:.3f} ± {std:.3f} dB"
    return f"{mean:.4e} ± {std:.4e}"


def print_summary(results: list[dict[str, float]], best_epochs: list[int]) -> None:
    ddof = 1 if len(results) > 1 else 0
    print(f"\nFINAL RESULTS over {len(results)} repeats (mean ± std)")
    for name in results[0]:
        values = np.asarray(
            [result[name] for result in results], dtype=np.float64
        )
        print(
            f"{name}: {format_summary_value(name, float(values.mean()), float(values.std(ddof=ddof)))}"
        )
    epochs = np.asarray(best_epochs, dtype=np.float64)
    print(
        f"best_epoch: {epochs.mean():.2f} ± {epochs.std(ddof=ddof):.2f}"
    )


def main() -> None:
    global SEED, REPEAT, TRAIN_RATIO, EPOCHS, HIDDEN, GCN_LAYERS
    global DROPOUT, LR, WEIGHT_DECAY, GRAD_CLIP, CACHE_GRAPHS, DEVICE, OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT,
                        help='Processed directory containing the three CSV subdirectories')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR, help='Checkpoint and metrics directory')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--seed', type=int, default=SEED, help='First random seed')
    parser.add_argument('--repeats', type=int, default=REPEAT, help='Independent training runs')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Epochs per run')
    parser.add_argument('--hidden', type=int, default=HIDDEN, help='Hidden channels')
    parser.add_argument('--layers', type=int, default=GCN_LAYERS, help='Propagation steps')
    parser.add_argument('--dropout', type=float, default=DROPOUT, help='Dropout probability')
    parser.add_argument('--lr', type=float, default=LR, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY, help='AdamW weight decay')
    parser.add_argument('--grad-clip', type=float, default=GRAD_CLIP, help='Gradient norm limit')
    parser.add_argument('--train-ratio', type=float, default=TRAIN_RATIO, help='Training fraction')
    parser.add_argument('--no-cache', action='store_true', help='Read and build graphs on demand')
    args = parser.parse_args()
    if min(args.repeats, args.epochs, args.hidden, args.layers) < 1:
        parser.error('repeats, epochs, hidden, and layers must be positive')
    if not 0 < args.train_ratio < 1 or not 0 <= args.dropout < 1:
        parser.error('train-ratio must be in (0, 1); dropout must be in [0, 1)')
    if args.lr <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        parser.error('lr and grad-clip must be positive; weight-decay must be nonnegative')
    device_name = ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    try:
        DEVICE = torch.device(device_name)
        if DEVICE.type not in ('cpu', 'cuda'):
            raise ValueError('Only CPU and CUDA devices are supported')
        if DEVICE.type == 'cuda':
            if not torch.cuda.is_available():
                raise ValueError('CUDA is unavailable; use --device cpu or install a CUDA-enabled PyTorch build')
            torch.cuda.set_device(DEVICE)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    SEED, REPEAT, EPOCHS = args.seed, args.repeats, args.epochs
    HIDDEN, GCN_LAYERS, DROPOUT = args.hidden, args.layers, args.dropout
    LR, WEIGHT_DECAY, GRAD_CLIP = args.lr, args.weight_decay, args.grad_clip
    TRAIN_RATIO, CACHE_GRAPHS = args.train_ratio, not args.no_cache
    OUTPUT_DIR = args.output_dir
    ids, files = discover_samples(args.data_root)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config['resolved_device'] = str(DEVICE)
    (OUTPUT_DIR / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    torch.set_float32_matmul_precision('high')
    results, best_epochs = [], []
    for repeat_index in range(1, REPEAT + 1):
        metrics, best_epoch = run_experiment(repeat_index, SEED + repeat_index - 1, ids, files)
        results.append(metrics)
        best_epochs.append(best_epoch)
        (OUTPUT_DIR / 'metrics.json').write_text(
            json.dumps({'results': results, 'best_epochs': best_epochs}, indent=2), encoding='utf-8')
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    print_summary(results, best_epochs)


if __name__ == '__main__':
    main()


