"""Utility script to export learned embeddings from a trained MENTOR checkpoint.

This script reconstructs the dataset and model configuration, loads the
specified checkpoint, materialises the lazily-built embedding tensors, and
writes the user/item embeddings (including modality-specific variants) together
with their aligned ids as ``.npy`` files. The default behaviour mirrors the
training setup used by ``main.py`` – it expects the same dataset artefacts to be
available locally and will look for checkpoints under ``saved/`` inside the
``src`` directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

import numpy as np
import torch

from utils_package.configurator import Config
from utils_package.dataset import RecDataset
from utils_package.dataloader import TrainDataLoader
from utils_package.utils import get_model, init_seed


_IGNORED_STATE_SUFFIXES: Tuple[str, ...] = ("v_preference", "t_preference", "id_preference")


def _default_checkpoint_path(src_dir: Path, model: str, dataset: str, checkpoint_dir: str) -> Path:
    """Return the default location of the checkpoint file."""

    return src_dir / checkpoint_dir / f"{model}-{dataset}-best.pth"


def _default_output_dir(src_dir: Path, model: str, dataset: str) -> Path:
    """Return the default directory to save exported embeddings."""

    return src_dir / "exported_embeddings" / f"{model}-{dataset}"


def _prepare_model(config: Config) -> Tuple[torch.nn.Module, TrainDataLoader]:
    """Re-create the dataloader and model exactly as during training."""

    dataset = RecDataset(config)
    train_dataset, _, _ = dataset.split()

    # Some downstream utilities expect these attributes to be populated on the
    # dataset objects returned by ``split``.
    if hasattr(train_dataset, "df"):
        train_dataset.inter_num = len(train_dataset.df)  # type: ignore[attr-defined]
    else:
        try:
            train_dataset.inter_num = len(train_dataset)  # type: ignore[attr-defined]
        except TypeError:
            pass

    train_loader = TrainDataLoader(
        config,
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=False,
    )

    model_cls = get_model(config["model"])
    if "device_torch" in config and isinstance(config["device_torch"], torch.device):
        device = config["device_torch"]
    else:
        device = torch.device(config["device"])
    model = model_cls(config, train_loader).to(device)
    model.eval()
    return model, train_loader


def _load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: Union[str, torch.device],
) -> None:
    """Load parameters from ``checkpoint_path`` into ``model``."""

    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    filtered_state: Dict[str, torch.Tensor] = {}
    ignored_keys: Dict[str, torch.Size] = {}
    stripped_module_prefix = False
    for key, value in state_dict.items():
        normalised_key = key[7:] if key.startswith("module.") else key
        if normalised_key != key:
            stripped_module_prefix = True
        if _should_ignore_state_key(normalised_key):
            ignored_keys[normalised_key] = value.shape
            continue
        filtered_state[normalised_key] = value

    if stripped_module_prefix:
        print("Stripped 'module.' prefixes from DataParallel checkpoint keys.")
    if ignored_keys:
        print("Ignoring checkpoint keys:")
        for key, shape in ignored_keys.items():
            print(f"  - {key} (shape={tuple(shape)})")

    try:
        model.load_state_dict(filtered_state, strict=True)
    except RuntimeError as err:
        print("Strict checkpoint load failed; retrying with strict=False.")
        print(f"  Reason: {err}")
        incompatibilities = model.load_state_dict(filtered_state, strict=False)
        if incompatibilities.missing_keys:
            print("  Missing keys:")
            for key in incompatibilities.missing_keys:
                print(f"    - {key}")
        if incompatibilities.unexpected_keys:
            print("  Unexpected keys:")
            for key in incompatibilities.unexpected_keys:
                print(f"    - {key}")


def _split_embeddings(
    name: str,
    embedding: torch.Tensor,
    n_users: int,
    n_items: int,
    *,
    zero_nonfinite: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a concatenated [user; item] tensor into numpy arrays."""

    array = _tensor_to_numpy(name, embedding, zero_nonfinite=zero_nonfinite)
    if array.ndim != 2:
        raise ValueError(f"Embedding '{name}' is expected to be 2-D, found shape {array.shape}")

    expected_rows = n_users + n_items
    if array.shape[0] != expected_rows:
        raise ValueError(
            f"Embedding '{name}' has incompatible first dimension: "
            f"expected {expected_rows}, found {array.shape[0]}"
        )

    user_embed = array[:n_users]
    item_embed = array[n_users:]
    return user_embed, item_embed


def _export_embeddings(
    model: torch.nn.Module,
    output_dir: Path,
    *,
    zero_nonfinite: bool,
) -> Dict[str, Path]:
    """Write numpy arrays for all available embedding variants."""

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: Dict[str, Path] = {}

    if not hasattr(model, "n_users") or not hasattr(model, "n_items"):
        raise AttributeError("Model does not expose 'n_users'/'n_items' attributes needed for export")

    try:
        n_users = int(model.n_users)
        n_items = int(model.n_items)
    except TypeError as err:
        raise TypeError("Model 'n_users'/'n_items' attributes must be convertible to int") from err

    if not hasattr(model, "result_embed") or not isinstance(model.result_embed, torch.Tensor):
        raise AttributeError("Model is missing the mandatory 'result_embed' tensor after materialisation")

    # Core user/item representation used for scoring
    user_all, item_all = _split_embeddings(
        "result_embed", model.result_embed, n_users, n_items, zero_nonfinite=zero_nonfinite
    )
    saved_paths["user_all"] = _save_numpy(output_dir, "user_all.npy", user_all)
    saved_paths["item_all"] = _save_numpy(output_dir, "item_all.npy", item_all)

    # Modality-specific representations, if present in the checkpoint
    if getattr(model, "result_embed_v", None) is not None:
        user_v, item_v = _split_embeddings(
            "result_embed_v", model.result_embed_v, n_users, n_items, zero_nonfinite=zero_nonfinite
        )
        saved_paths["user_v"] = _save_numpy(output_dir, "user_v.npy", user_v)
        saved_paths["item_v"] = _save_numpy(output_dir, "item_v.npy", item_v)

    if getattr(model, "result_embed_t", None) is not None:
        user_t, item_t = _split_embeddings(
            "result_embed_t", model.result_embed_t, n_users, n_items, zero_nonfinite=zero_nonfinite
        )
        saved_paths["user_t"] = _save_numpy(output_dir, "user_t.npy", user_t)
        saved_paths["item_t"] = _save_numpy(output_dir, "item_t.npy", item_t)

    if getattr(model, "result_embed_guide", None) is not None:
        user_g, item_g = _split_embeddings(
            "result_embed_guide", model.result_embed_guide, n_users, n_items, zero_nonfinite=zero_nonfinite
        )
        saved_paths["user_guide"] = _save_numpy(output_dir, "user_guide.npy", user_g)
        saved_paths["item_guide"] = _save_numpy(output_dir, "item_guide.npy", item_g)

    # Write aligned identifier arrays expected by downstream consumers.
    saved_paths["user_id"] = _save_numpy(
        output_dir,
        "user_id.npy",
        np.arange(n_users, dtype=np.int64),
    )
    saved_paths["item_id"] = _save_numpy(
        output_dir,
        "item_id.npy",
        np.arange(n_items, dtype=np.int64),
    )

    return saved_paths


def _save_numpy(output_dir: Path, filename: str, array: np.ndarray) -> Path:
    """Persist ``array`` to ``output_dir/filename`` as ``.npy``."""

    target = output_dir / filename
    np.save(target, array)
    return target


def _tensor_to_numpy(name: str, tensor: torch.Tensor, *, zero_nonfinite: bool) -> np.ndarray:
    """Convert ``tensor`` to a float32 numpy array and validate its contents."""

    if tensor is None:
        raise ValueError(f"Tensor '{name}' is not present on the model (did forward() populate it?)")

    array = tensor.detach().to(torch.float32).cpu().numpy()
    if not np.all(np.isfinite(array)):
        if zero_nonfinite:
            print(f"Tensor '{name}' contains non-finite values; replacing them with zeros.")
            array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            raise ValueError(f"Non-finite values encountered in tensor '{name}'")
    return array


def _should_ignore_state_key(key: str) -> bool:
    """Return ``True`` if ``key`` should be ignored when loading checkpoints."""

    suffix = key.split(".")[-1]
    return suffix in _IGNORED_STATE_SUFFIXES


def _materialise_embeddings(model: torch.nn.Module, train_loader: TrainDataLoader) -> None:
    """Run the minimal forward pass needed to populate embedding attributes."""

    if hasattr(model, "pre_epoch_processing"):
        model.pre_epoch_processing()

    if hasattr(train_loader, "pretrain_setup"):
        train_loader.pretrain_setup()
    iterator = iter(train_loader)
    try:
        batch = next(iterator)
    except StopIteration as exc:
        raise RuntimeError("Training dataloader yielded no batches to materialise embeddings") from exc

    with torch.no_grad():
        candidates: Iterable[object] = [batch]
        if isinstance(batch, list):
            candidates = [batch, tuple(batch)]
        elif isinstance(batch, tuple):
            candidates = [batch, list(batch)]

        for candidate in candidates:
            try:
                model.forward(candidate)
                break
            except (TypeError, ValueError, RuntimeError, IndexError, AttributeError):
                continue
        else:
            reference_param = next(model.parameters(), None)
            if reference_param is None:
                raise RuntimeError("Model has no parameters to infer device for dummy forward pass")

            device = reference_param.device
            dummy_user = torch.zeros(1, dtype=torch.long, device=device)
            dummy_pos_item = torch.zeros(1, dtype=torch.long, device=device)
            try:
                item_count = int(model.n_items)
            except TypeError as err:
                raise TypeError("Model 'n_items' attribute must be convertible to int for dummy forward pass") from err
            neg_index = 1 if item_count > 1 else 0
            dummy_neg_item = torch.full((1,), neg_index, dtype=torch.long, device=device)

            dummy_inputs = [
                (dummy_user, dummy_pos_item, dummy_neg_item),
                [dummy_user, dummy_pos_item, dummy_neg_item],
            ]
            for candidate in dummy_inputs:
                try:
                    model.forward(candidate)
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError(
                    "Failed to materialise embeddings: the model rejected both a dataloader batch "
                    "and dummy (user, pos_item, neg_item) inputs"
                )

    if not hasattr(model, "result_embed") or not isinstance(model.result_embed, torch.Tensor):
        raise AttributeError("Model forward pass did not populate 'result_embed'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export user/item embeddings from a trained checkpoint.")
    parser.add_argument("--model", default="MENTOR", help="Model name to load (default: MENTOR)")
    parser.add_argument("--dataset", default="baby", help="Dataset name to load (default: baby)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint (.pth) file. Defaults to saved/{model}-{dataset}-best.pth.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store exported embeddings. Defaults to src/exported_embeddings/{model}-{dataset}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed override (uses the value from the config by default).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the device to run on (e.g. 'cpu' or 'cuda:0').",
    )
    parser.add_argument(
        "--zero-nonfinite",
        action="store_true",
        help="Replace NaN/Inf values in exported tensors with zeros instead of failing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    src_dir = Path(__file__).resolve().parent

    config_overrides: Dict[str, object] = {}
    if args.seed is not None:
        config_overrides["seed"] = [args.seed]

    config = Config(args.model, args.dataset, config_overrides)

    base_device = config["device"]
    if isinstance(base_device, torch.device):
        config["device_torch"] = base_device
        config["device"] = str(base_device)

    # ``Config`` stores seed as a list to enable hyper-parameter search; reuse the first value.
    seed_value = config["seed"][0] if isinstance(config["seed"], (list, tuple)) else config["seed"]
    init_seed(seed_value)

    if args.device is not None:
        override_device = torch.device(args.device)
        config["device_torch"] = override_device
        config["device"] = str(override_device)
        config["use_gpu"] = override_device.type == "cuda"
        if override_device.type == "cuda" and override_device.index is not None:
            config["gpu_id"] = override_device.index

    model, train_loader = _prepare_model(config)

    checkpoint_dir = (
        config["checkpoint_dir"]
        if "checkpoint_dir" in config and config["checkpoint_dir"]
        else "saved"
    )
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else _default_checkpoint_path(src_dir, config["model"], config["dataset"], checkpoint_dir)
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    map_location: Union[str, torch.device]
    if "device_torch" in config and isinstance(config["device_torch"], torch.device):
        map_location = config["device_torch"]
    else:
        map_location = str(config["device"])

    _load_checkpoint(model, checkpoint_path, map_location)

    _materialise_embeddings(model, train_loader)

    output_dir = Path(args.output_dir) if args.output_dir is not None else _default_output_dir(src_dir, config["model"], config["dataset"])

    saved_files = _export_embeddings(model, output_dir, zero_nonfinite=args.zero_nonfinite)

    print("Exported embeddings:")
    for name, path in saved_files.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

