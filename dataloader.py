"""CSV loading, training-only normalization, and finite element graph construction."""
from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass

import torch
from typing import Protocol

import numpy as np
import pandas as pd
from tqdm import tqdm


DATA_ROOT = Path(__file__).resolve().parent / "datasets" / "elbow_bracket" / "elbow_bracket_processed"
COORD_COLS = ["x", "y", "z"]
STRESS_COLS = ["von_mises"]
NODE_FEATURE_DIM = 4

# Abaqus C3D10M node ordering. Nodes 1..4 are the tetrahedron vertices and
# nodes 5..10 are the midside nodes on edges 1-2, 2-3, 3-1, 1-4, 2-4, 3-4.
# Each quadratic edge is represented by its two physical half-edge segments.
#
C3D10M_EDGE_SEGMENTS = (
    (0, 4),
    (4, 1),
    (1, 5),
    (5, 2),
    (2, 6),
    (6, 0),
    (0, 7),
    (7, 3),
    (1, 8),
    (8, 3),
    (2, 9),
    (9, 3),
)

NODE_COLUMN_PATTERN = re.compile(r"^node(\d+)$", flags=re.IGNORECASE)


class GraphDataset(Protocol):
    cache_enabled: bool

    def __len__(self) -> int: ...

    def __getitem__(self, index: int): ...


def require_columns(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    missing = [column for column in cols if column not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")


def indexed_files(folder: Path, prefix: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.csv$")
    for path in folder.glob(f"{prefix}_*.csv"):
        match = pattern.match(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def discover_samples(root: Path) -> tuple[list[int], dict[str, dict[int, Path]]]:
    print(f"Scanning dataset: {root}")
    categories = (
        ("coord", root / "input_coord", "coord"),
        ("matrix", root / "input_matrix", "matrix"),
        ("stress", root / "output_stress", "stress"),
    )
    files: dict[str, dict[int, Path]] = {}
    for name, folder, prefix in tqdm(categories, desc="Scanning directories", unit="category"):
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {folder}")
        files[name] = indexed_files(folder, prefix)

    ids = sorted(set.intersection(*(set(paths) for paths in files.values())))
    if len(ids) < 2:
        raise RuntimeError("At least two complete samples with matching coordinate, connectivity, and stress IDs are required.")

    mismatches = {
        name: sorted(set(paths).symmetric_difference(ids))
        for name, paths in files.items()
        if set(paths) != set(ids)
    }
    if mismatches:
        raise RuntimeError(f"Sample IDs differ between CSV categories: {mismatches}")

    print(f"Dataset scan complete: {len(ids)} complete samples; fixed loads are not input features.")
    return ids, files


class RunningStats:
    def __init__(self, dim: int) -> None:
        self.n = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sumsq = np.zeros(dim, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        self.n += values.shape[0]
        self.sum += values.sum(axis=0)
        self.sumsq += np.square(values).sum(axis=0)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.n == 0:
            raise RuntimeError("Cannot compute statistics from empty data.")
        mean = self.sum / self.n
        variance = np.maximum(self.sumsq / self.n - np.square(mean), 0.0)
        std = np.sqrt(variance)
        std[std < 1e-12] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)


def load_node_arrays(
    sample_id: int,
    files: dict[str, dict[int, Path]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    coord_path = files["coord"][sample_id]
    stress_path = files["stress"][sample_id]
    coord_df = pd.read_csv(coord_path, usecols=["node", *COORD_COLS])
    stress_df = pd.read_csv(stress_path, usecols=["node", *STRESS_COLS])
    require_columns(coord_df, ["node", *COORD_COLS], coord_path)
    require_columns(stress_df, ["node", *STRESS_COLS], stress_path)

    node_ids = coord_df["node"].to_numpy(dtype=np.int64, copy=False)
    if len(np.unique(node_ids)) != len(node_ids):
        raise ValueError(f"{coord_path} contains duplicate node labels.")
    node_to_index = {int(node): index for index, node in enumerate(node_ids)}
    coords = coord_df[COORD_COLS].to_numpy(dtype=np.float32)

    if stress_df["node"].duplicated().any():
        raise ValueError(f"{stress_path} contains duplicate node labels.")
    stress_table = stress_df.set_index(stress_df["node"].astype(np.int64))
    missing = np.setdiff1d(node_ids, stress_table.index.to_numpy(dtype=np.int64))
    if missing.size:
        raise ValueError(f"{stress_path} is missing stress values for {missing.size} nodes.")
    stress = stress_table.loc[node_ids, STRESS_COLS].to_numpy(dtype=np.float32)
    return node_ids, coords, stress, node_to_index


def collect_train_stats(
    train_ids: list[int],
    files: dict[str, dict[int, Path]],
) -> dict[str, np.ndarray]:
    print("Computing coordinate and stress statistics from training samples.")
    coord_stats = RunningStats(3)
    stress_stats = RunningStats(1)
    for sample_id in tqdm(train_ids, desc="Training statistics", unit="sample"):
        _, coords, stress, _ = load_node_arrays(sample_id, files)
        coord_stats.update(coords)
        stress_stats.update(stress)
    coord_mean, coord_std = coord_stats.finalize()
    stress_mean, stress_std = stress_stats.finalize()
    print("Training statistics complete.")
    return {
        "coord_mean": coord_mean,
        "coord_std": coord_std,
        "stress_mean": stress_mean,
        "stress_std": stress_std,
    }


def discover_element_node_columns(matrix_path: Path) -> list[str]:
    """Find and numerically sort node1 ... nodeN connectivity columns."""
    header = pd.read_csv(matrix_path, nrows=0)
    require_columns(header, ["element"], matrix_path)

    numbered_columns: list[tuple[int, str]] = []
    seen_numbers: set[int] = set()
    for column in header.columns:
        match = NODE_COLUMN_PATTERN.match(str(column))
        if not match:
            continue

        number = int(match.group(1))
        if number <= 0:
            raise ValueError(f"{matrix_path} node column numbers must start at 1: {column}")
        if number in seen_numbers:
            raise ValueError(
                f"{matrix_path} contains duplicate node column number node{number}."
            )
        seen_numbers.add(number)
        numbered_columns.append((number, str(column)))

    if not numbered_columns:
        raise ValueError(
            f"{matrix_path} contains no node1...nodeN connectivity columns."
        )

    numbered_columns.sort(key=lambda item: item[0])
    return [column for _, column in numbered_columns]


def _encode_undirected_pairs(
    first: np.ndarray,
    second: np.ndarray,
    node_count: int,
) -> np.ndarray:
    """Encode undirected non-self edges as integers for deduplication."""
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    if first.shape != second.shape:
        raise ValueError("Edge endpoint arrays must have matching shapes.")

    low = np.minimum(first, second)
    high = np.maximum(first, second)

    keep = low != high
    low = low[keep]
    high = high[keep]
    if low.size == 0:
        return np.empty(0, dtype=np.int64)

    return low * node_count + high


def _complete_element_edges(
    mapped_nodes: np.ndarray,
    node_count: int,
) -> np.ndarray:
    """Connect every distinct pair of valid nodes within an element."""
    mapped_nodes = np.asarray(mapped_nodes, dtype=np.int64)

    if mapped_nodes.size > 1:
        _, first_positions = np.unique(mapped_nodes, return_index=True)
        mapped_nodes = mapped_nodes[np.sort(first_positions)]

    count = mapped_nodes.size
    if count < 2:
        return np.empty(0, dtype=np.int64)

    first_pos, second_pos = np.triu_indices(count, k=1)
    return _encode_undirected_pairs(
        mapped_nodes[first_pos],
        mapped_nodes[second_pos],
        node_count,
    )


def _c3d10m_element_edges(
    mapped_nodes: np.ndarray,
    node_count: int,
) -> np.ndarray:
    """Construct the twelve physical half-edge segments of a C3D10M element."""
    mapped_nodes = np.asarray(mapped_nodes, dtype=np.int64)
    if mapped_nodes.size != 10:
        raise ValueError("C3D10M elements require exactly 10 nodes.")

    segments = np.asarray(C3D10M_EDGE_SEGMENTS, dtype=np.int64)
    return _encode_undirected_pairs(
        mapped_nodes[segments[:, 0]],
        mapped_nodes[segments[:, 1]],
        node_count,
    )


def build_polyhedron_edges(
    matrix_path: Path,
    node_to_index: dict[int, int],
    node_count: int,
) -> tuple[np.ndarray, "torch.Tensor", np.ndarray]:
    """Build unique undirected pairs, bidirectional edge indices, and node degrees.

    Complete node1..node10 rows use the C3D10M half-edge template. Other
    elements use complete within-element connectivity, not inferred physical
    edges. Blank node entries are permitted; invalid labels are rejected."""
    # Local import keeps simple file-discovery/statistics checks independent of
    # torch initialization while training scripts still receive torch tensors.
    import torch

    if not node_to_index:
        raise ValueError(f"{matrix_path} has no nodes in the coordinate file.")
    if node_count <= 0:
        raise ValueError("node_count must be positive.")
    if max(node_to_index.values(), default=-1) >= node_count:
        raise ValueError(
            "Internal node index exceeds node_count."
        )
    if min(node_to_index.values(), default=0) < 0:
        raise ValueError("Internal node indices cannot be negative.")

    node_columns = discover_element_node_columns(matrix_path)
    matrix_df = pd.read_csv(matrix_path, usecols=["element", *node_columns])
    require_columns(matrix_df, ["element", *node_columns], matrix_path)

    raw_nodes = matrix_df[node_columns]

    numeric_nodes = raw_nodes.apply(pd.to_numeric, errors="coerce")
    invalid_numeric = raw_nodes.notna() & numeric_nodes.isna()
    if invalid_numeric.to_numpy().any():
        row_pos, col_pos = np.argwhere(invalid_numeric.to_numpy())[0]
        column = node_columns[int(col_pos)]
        element = matrix_df.iloc[int(row_pos)]["element"]
        value = raw_nodes.iloc[int(row_pos), int(col_pos)]
        raise ValueError(
            f"{matrix_path} element={element}, {column} contains a nonnumeric node label: {value!r}"
        )

    node_column_numbers = np.asarray(
        [int(NODE_COLUMN_PATTERN.match(column).group(1)) for column in node_columns],
        dtype=np.int64,
    )

    encoded_edges: list[np.ndarray] = []
    values = numeric_nodes.to_numpy(dtype=np.float64)

    for row_index, row in enumerate(values):
        valid_mask = ~np.isnan(row)
        valid_values = row[valid_mask]
        valid_column_numbers = node_column_numbers[valid_mask]

        element = matrix_df.iloc[row_index]["element"]
        if valid_values.size < 2:
            raise ValueError(
                f"{matrix_path} element={element} has fewer than two valid nodes."
            )

        if np.any(~np.isfinite(valid_values)):
            raise ValueError(
                f"{matrix_path} element={element} contains a nonfinite node label."
            )
        if np.any(valid_values < 0):
            raise ValueError(
                f"{matrix_path} element={element} contains a negative node label."
            )
        if np.any(valid_values != np.floor(valid_values)):
            raise ValueError(
                f"{matrix_path} element={element} contains a noninteger node label."
            )

        labels = valid_values.astype(np.int64)

        mapped = np.fromiter(
            (node_to_index.get(int(label), -1) for label in labels),
            dtype=np.int64,
            count=labels.size,
        )
        missing_mask = mapped < 0
        if np.any(missing_mask):
            missing = np.unique(labels[missing_mask])
            preview = missing[:10].tolist()
            suffix = "..." if missing.size > 10 else ""
            raise ValueError(
                f"{matrix_path} element={element} has {missing.size} nodes "
                f"missing from the coordinate file: {preview}{suffix}"
            )

        is_c3d10m = (
            mapped.size == 10
            and np.array_equal(valid_column_numbers, np.arange(1, 11))
        )

        if is_c3d10m:
            encoded = _c3d10m_element_edges(mapped, node_count)
        else:
            encoded = _complete_element_edges(mapped, node_count)

        if encoded.size:
            encoded_edges.append(encoded)

    if encoded_edges:
        unique_edges = np.unique(np.concatenate(encoded_edges))
        pairs = np.column_stack(
            (unique_edges // node_count, unique_edges % node_count)
        ).astype(np.int64, copy=False)
    else:
        pairs = np.empty((0, 2), dtype=np.int64)

    if pairs.size:
        src = np.concatenate((pairs[:, 0], pairs[:, 1]))
        dst = np.concatenate((pairs[:, 1], pairs[:, 0]))
        edge_index = torch.from_numpy(np.stack((src, dst), axis=0)).long()
        degree = np.bincount(src, minlength=node_count).astype(np.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        degree = np.zeros(node_count, dtype=np.float32)

    return pairs, edge_index, degree


def build_c3d10m_edges(
    matrix_path: Path,
    node_to_index: dict[int, int],
    node_count: int,
) -> tuple[np.ndarray, "torch.Tensor", np.ndarray]:
    """Compatibility alias for build_polyhedron_edges."""
    return build_polyhedron_edges(matrix_path, node_to_index, node_count)


def make_node_features(
    coords: np.ndarray,
    degree: np.ndarray,
    stats: dict[str, np.ndarray],
) -> np.ndarray:
    coord_feature = (coords - stats["coord_mean"]) / stats["coord_std"]
    degree_feature = np.log1p(degree).reshape(-1, 1)
    degree_feature = (degree_feature - degree_feature.mean()) / (
        degree_feature.std() + 1e-6
    )
    return np.concatenate((coord_feature, degree_feature), axis=1).astype(
        np.float32, copy=False
    )


def preload_dataset(dataset: GraphDataset, label: str) -> None:
    if not dataset.cache_enabled:
        print(f"{label}: graph cache disabled; loading on demand.")
        return
    print(f"Loading and normalizing {label}...")
    for index in tqdm(range(len(dataset)), desc=f"Loading {label}", unit="sample"):
        dataset[index]
    print(f"{label}: loading complete.")


def normalized_adjacency(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """Build symmetric normalized adjacency without adding self-loops."""
    if edge_index.shape[1] == 0:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty(0, dtype=torch.float32),
            (n, n),
        ).coalesce()

    value = torch.ones(edge_index.shape[1], dtype=torch.float32)
    degree = torch.zeros(n, dtype=torch.float32)
    degree.scatter_add_(0, edge_index[0], value)
    norm = value * degree[edge_index[0]].rsqrt() * degree[edge_index[1]].rsqrt()
    return torch.sparse_coo_tensor(edge_index, norm, (n, n)).coalesce()


def normalized_laplacian(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    if edge_index.shape[1] == 0:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty(0),
            (n, n),
        ).coalesce()
    value = torch.ones(edge_index.shape[1], dtype=torch.float32)
    degree = torch.zeros(n, dtype=torch.float32)
    degree.scatter_add_(0, edge_index[0], value)
    norm = value * degree[edge_index[0]].rsqrt() * degree[edge_index[1]].rsqrt()
    active = torch.nonzero(degree > 0, as_tuple=False).flatten()
    diag_index = torch.stack([active, active], dim=0)
    index = torch.cat([diag_index, edge_index], dim=1)
    lap_value = torch.cat([torch.ones(active.numel()), -norm], dim=0)
    return torch.sparse_coo_tensor(index, lap_value, (n, n)).coalesce()


def build_topology(
    matrix_path: Path,
    node_to_index: dict[int, int],
    n: int,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    pairs, edge_index, degree = build_c3d10m_edges(matrix_path, node_to_index, n)
    return (
        normalized_adjacency(edge_index, n),
        normalized_laplacian(edge_index, n),
        degree,
        pairs,
    )


@dataclass
class GraphData:
    sample_id: int
    x: torch.Tensor
    y: torch.Tensor
    adj: torch.Tensor
    lap: torch.Tensor
    coords: torch.Tensor
    edges: torch.Tensor


class FEMGraphDataset:
    def __init__(
        self,
        ids: list[int],
        files: dict[str, dict[int, Path]],
        stats: dict[str, np.ndarray],
        cache: bool = True,
    ) -> None:
        self.ids = ids
        self.files = files
        self.stats = stats
        self.cache_enabled = cache
        self.cache = {}

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> GraphData:
        if index in self.cache:
            return self.cache[index]
        sample_id = self.ids[index]
        _, coords, stress, node_to_index = load_node_arrays(
            sample_id, self.files
        )
        adj, lap, degree, edges = build_topology(
            self.files["matrix"][sample_id], node_to_index, len(coords)
        )
        x = make_node_features(coords, degree, self.stats)
        graph = GraphData(
            sample_id,
            torch.from_numpy(x).float(),
            torch.from_numpy(stress).float(),
            adj,
            lap,
            torch.from_numpy(coords).float(),
            torch.from_numpy(edges).long(),
        )
        if self.cache_enabled:
            self.cache[index] = graph
        return graph


