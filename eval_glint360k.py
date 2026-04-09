import argparse
import json
import pickle
import tarfile
from pathlib import Path
from typing import Callable

import cv2
import matplotlib
import numpy as np
import torch

from backbones import get_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_SIZE = (112, 112)
FAR_TARGETS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
INDEX_VERSION = 1


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate TransFace on Glint360K WebDataset shards.")
    parser.add_argument("--model-prefix", required=True, help="Path to the backbone checkpoint.")
    parser.add_argument("--data-root", required=True, type=str, help="Path to the Glint360K tar shards.")
    parser.add_argument("--result-dir", required=True, type=str, help="Directory for evaluation outputs.")
    parser.add_argument("--network", required=True, type=str, help="Backbone network name.")
    parser.add_argument("--batch-size", default=128, type=int, help="Inference batch size.")
    parser.add_argument("--job", default="glint360k_eval", type=str, help="Output subdirectory name.")
    parser.add_argument(
        "--index-cache",
        default=None,
        type=str,
        help="Pickle cache for the Glint360K identity index. Defaults to <result-dir>/glint360k_index.pkl.",
    )
    parser.add_argument(
        "--num-negatives-per-id",
        default=32,
        type=int,
        help="Number of deterministic negative gallery matches sampled for each probe.",
    )
    parser.add_argument("--seed", default=42, type=int, help="Reserved for deterministic evaluation configuration.")
    parser.add_argument("--no-flip-test", action="store_true", help="Disable horizontal flip test.")
    parser.add_argument(
        "--chunk-size",
        default=32768,
        type=int,
        help="Chunk size used during negative score computation.",
    )
    return parser


def identity_sort_key(identity: str) -> tuple[int, object]:
    if identity.isdigit():
        return (0, int(identity))
    return (1, identity)


def find_tar_files(data_root: Path) -> list[Path]:
    tar_files = sorted(data_root.glob("glint360k_train-*.tar"))
    if not tar_files:
        raise FileNotFoundError(f"No Glint360K tar shards found under {data_root}")
    return tar_files


def default_index_cache_path(result_dir: Path) -> Path:
    return result_dir / "glint360k_index.pkl"


def _merge_first_two_refs(existing_refs: list[tuple[str, str, int, int]], new_refs: list[tuple[str, str, int, int]]) -> list[tuple[str, str, int, int]]:
    merged = sorted(existing_refs + new_refs, key=lambda ref: (ref[0], ref[1]))
    return merged[:2]


def build_glint_index(data_root: Path) -> dict[str, object]:
    identity_to_refs: dict[str, list[tuple[str, str, int, int]]] = {}
    identity_counts: dict[str, int] = {}
    tar_files = find_tar_files(data_root)

    for tar_index, tar_path in enumerate(tar_files, start=1):
        print(f"[index] scanning {tar_index}/{len(tar_files)}: {tar_path.name}")
        tar_identity_refs: dict[str, list[tuple[str, str, int, int]]] = {}
        pending_labels: dict[str, str] = {}
        pending_jpegs: dict[str, tuple[str, str, int, int]] = {}

        with tarfile.open(tar_path, "r") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                suffix = Path(member.name).suffix.lower()
                if suffix == ".cls":
                    key = Path(member.name).stem
                    cls_file = tar.extractfile(member)
                    if cls_file is None:
                        continue
                    identity = cls_file.read().decode("utf-8", "ignore").strip()
                    if not identity:
                        continue
                    pending_labels[key] = identity
                    pending_ref = pending_jpegs.pop(key, None)
                    if pending_ref is not None:
                        tar_identity_refs.setdefault(identity, []).append(pending_ref)
                elif suffix == ".jpg":
                    key = Path(member.name).stem
                    ref = (tar_path.name, member.name, member.offset_data, member.size)
                    identity = pending_labels.pop(key, None)
                    if identity is None:
                        pending_jpegs[key] = ref
                    else:
                        tar_identity_refs.setdefault(identity, []).append(ref)

        for identity, refs in tar_identity_refs.items():
            refs.sort(key=lambda ref: ref[1])
            identity_to_refs[identity] = _merge_first_two_refs(identity_to_refs.get(identity, []), refs)
            identity_counts[identity] = min(identity_counts.get(identity, 0) + len(refs), 2)

    eligible_id_list = sorted(
        [identity for identity, count in identity_counts.items() if count >= 2],
        key=identity_sort_key,
    )
    return {
        "version": INDEX_VERSION,
        "identity_to_refs": identity_to_refs,
        "eligible_id_list": eligible_id_list,
    }


def load_or_build_index(data_root: Path, cache_path: Path) -> dict[str, object]:
    if cache_path.exists():
        print(f"[index] loading cache: {cache_path}")
        with cache_path.open("rb") as handle:
            cache = pickle.load(handle)
        if cache.get("version") != INDEX_VERSION:
            print("[index] cache version mismatch, rebuilding")
        else:
            return cache

    cache = build_glint_index(data_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[index] wrote cache: {cache_path}")
    return cache


def build_reference_lists(index_cache: dict[str, object]) -> tuple[list[str], list[tuple[str, str, int, int]], list[tuple[str, str, int, int]]]:
    identity_to_refs = index_cache["identity_to_refs"]
    eligible_id_list = index_cache["eligible_id_list"]
    gallery_refs = []
    probe_refs = []
    filtered_ids = []

    for identity in eligible_id_list:
        refs = identity_to_refs.get(identity, [])
        if len(refs) < 2:
            continue
        filtered_ids.append(identity)
        gallery_refs.append(refs[0])
        probe_refs.append(refs[1])

    return filtered_ids, gallery_refs, probe_refs


def _read_tar_bytes(data_root: Path, ref: tuple[str, str, int, int]) -> bytes:
    tar_name, _member_name, offset, size = ref
    tar_path = data_root / tar_name
    with tar_path.open("rb") as handle:
        handle.seek(offset)
        return handle.read(size)


def decode_reference_image(data_root: Path, ref: tuple[str, str, int, int]) -> np.ndarray | None:
    img_bytes = _read_tar_bytes(data_root, ref)
    array = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        return None
    if image.shape[0:2] != IMAGE_SIZE:
        image = cv2.resize(image, IMAGE_SIZE[::-1], interpolation=cv2.INTER_LINEAR)
    return image


def prepare_image_tensor(image_bgr: np.ndarray, flip: bool) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if flip:
        image_rgb = np.fliplr(image_rgb)
    chw = np.transpose(image_rgb, (2, 0, 1)).astype(np.float32)
    chw = (chw / 255.0 - 0.5) / 0.5
    return chw


def _normalize_features(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return features / norms


def _extract_feature_tensor(output: object) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        feature = output[0]
    else:
        feature = output
    if not isinstance(feature, torch.Tensor):
        raise TypeError(f"Unexpected model output type: {type(feature)!r}")
    return feature


def _decode_from_handle(handle: object, ref: tuple[str, str, int, int]) -> np.ndarray | None:
    _tar_name, _member_name, offset, size = ref
    handle.seek(offset)
    img_bytes = handle.read(size)
    array = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        return None
    if image.shape[0:2] != IMAGE_SIZE:
        image = cv2.resize(image, IMAGE_SIZE[::-1], interpolation=cv2.INTER_LINEAR)
    return image


def load_backbone_model(model_path: Path, network: str) -> torch.nn.Module:
    weight = torch.load(model_path, map_location="cpu")
    model = get_model(network, dropout=0, fp16=False).cuda()

    try:
        model.load_state_dict(weight)
    except RuntimeError:
        if all(key.startswith("module.") for key in weight.keys()):
            stripped = {key[len("module."):]: value for key, value in weight.items()}
            model.load_state_dict(stripped)
        else:
            wrapped = {f"module.{key}": value for key, value in weight.items()}
            model = torch.nn.DataParallel(model)
            model.load_state_dict(wrapped)
            model.eval()
            return model

    model = torch.nn.DataParallel(model)
    model.eval()
    return model


@torch.no_grad()
def extract_feature_matrix(
    refs: list[tuple[str, str, int, int]],
    data_root: Path,
    model_path: Path,
    network: str,
    batch_size: int,
    use_flip_test: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    model = load_backbone_model(model_path, network)
    num_refs = len(refs)
    features = None
    valid_mask = np.zeros(num_refs, dtype=bool)
    invalid_images = 0
    batch_tensors: list[np.ndarray] = []
    batch_indices: list[int] = []

    def flush_batch() -> None:
        nonlocal batch_tensors, batch_indices, features
        if not batch_tensors:
            return
        inputs = torch.from_numpy(np.stack(batch_tensors, axis=0)).cuda(non_blocking=True)
        batch_output = _extract_feature_tensor(model(inputs)).detach().cpu().numpy()
        if use_flip_test:
            embedding_dim = batch_output.shape[1]
            batch_output = batch_output.reshape(len(batch_indices), 2, embedding_dim).sum(axis=1)
        batch_output = _normalize_features(batch_output.astype(np.float32, copy=False))
        if features is None:
            features = np.zeros((num_refs, batch_output.shape[1]), dtype=np.float32)
        features[np.asarray(batch_indices)] = batch_output
        valid_mask[np.asarray(batch_indices)] = True
        batch_tensors = []
        batch_indices = []

    refs_by_tar: dict[str, list[tuple[int, tuple[str, str, int, int]]]] = {}
    for ref_index, ref in enumerate(refs):
        refs_by_tar.setdefault(ref[0], []).append((ref_index, ref))

    for tar_name in sorted(refs_by_tar):
        tar_path = data_root / tar_name
        tar_refs = sorted(refs_by_tar[tar_name], key=lambda item: item[1][2])
        print(f"[feature] {tar_name}: {len(tar_refs)} selected images")
        with tar_path.open("rb") as handle:
            for ref_index, ref in tar_refs:
                image = _decode_from_handle(handle, ref)
                if image is None:
                    invalid_images += 1
                    continue
                batch_tensors.append(prepare_image_tensor(image, flip=False))
                if use_flip_test:
                    batch_tensors.append(prepare_image_tensor(image, flip=True))
                batch_indices.append(ref_index)
                if len(batch_indices) == batch_size:
                    flush_batch()

    flush_batch()

    if features is None:
        features = np.zeros((num_refs, 0), dtype=np.float32)
    return features, valid_mask, invalid_images


def build_negative_gallery_indices(num_identities: int, num_negatives_per_id: int) -> np.ndarray:
    if num_identities < 2:
        raise ValueError("At least two eligible identities are required for negative pairs.")
    if num_negatives_per_id < 1:
        raise ValueError("--num-negatives-per-id must be at least 1.")
    max_negatives = min(num_negatives_per_id, num_identities - 1)
    offsets = np.arange(1, max_negatives + 1, dtype=np.int64)
    base = np.arange(num_identities, dtype=np.int64)[:, None]
    return (base + offsets[None, :]) % num_identities


def compute_scores(
    gallery_features: np.ndarray,
    probe_features: np.ndarray,
    num_negatives_per_id: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_identities = gallery_features.shape[0]
    negative_indices = build_negative_gallery_indices(num_identities, num_negatives_per_id)
    actual_negatives = negative_indices.shape[1]

    total_scores = num_identities + num_identities * actual_negatives
    scores = np.empty(total_scores, dtype=np.float32)
    labels = np.empty(total_scores, dtype=np.uint8)

    positive_scores = np.sum(probe_features * gallery_features, axis=1, dtype=np.float32)
    scores[:num_identities] = positive_scores
    labels[:num_identities] = 1

    cursor = num_identities
    offsets = np.arange(1, actual_negatives + 1, dtype=np.int64)
    for start in range(0, num_identities, chunk_size):
        end = min(start + chunk_size, num_identities)
        probe_chunk = probe_features[start:end]
        chunk_base = np.arange(start, end, dtype=np.int64)[:, None]
        chunk_indices = (chunk_base + offsets[None, :]) % num_identities
        negative_scores = np.sum(
            probe_chunk[:, None, :] * gallery_features[chunk_indices],
            axis=2,
            dtype=np.float32,
        ).reshape(-1)
        next_cursor = cursor + negative_scores.shape[0]
        scores[cursor:next_cursor] = negative_scores
        labels[cursor:next_cursor] = 0
        cursor = next_cursor

    return scores, labels


def compute_roc_metrics(scores: np.ndarray, labels: np.ndarray, far_targets: tuple[float, ...] = FAR_TARGETS) -> tuple[dict[str, float], np.ndarray, np.ndarray, float]:
    labels_bool = labels.astype(bool)
    num_positive = int(labels_bool.sum())
    num_negative = int((~labels_bool).sum())
    if num_positive == 0 or num_negative == 0:
        raise ValueError("ROC metrics require both positive and negative pairs.")

    order = np.argsort(scores)[::-1]
    sorted_labels = labels_bool[order]
    true_positive = np.cumsum(sorted_labels, dtype=np.int64)
    false_positive = np.cumsum(~sorted_labels, dtype=np.int64)

    tpr = np.concatenate(([0.0], true_positive / num_positive))
    fpr = np.concatenate(([0.0], false_positive / num_negative))
    auc_value = float(np.trapz(tpr, fpr))

    tar_by_far = {}
    for far in far_targets:
        index = np.searchsorted(fpr, far, side="right") - 1
        index = max(index, 0)
        tar_by_far[str(far)] = float(tpr[index])

    return tar_by_far, fpr, tpr, auc_value


def compute_verification_accuracy(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    labels_bool = labels.astype(bool)
    total_count = int(labels_bool.shape[0])
    if total_count == 0:
        raise ValueError("Accuracy requires at least one score.")

    num_positive = int(labels_bool.sum())
    num_negative = total_count - num_positive
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels_bool[order]
    true_positive = np.cumsum(sorted_labels, dtype=np.int64)
    false_positive = np.cumsum(~sorted_labels, dtype=np.int64)

    best_accuracy = num_negative / total_count
    best_threshold = float(np.nextafter(sorted_scores[0], np.inf))

    group_end_mask = np.ones(sorted_scores.shape[0], dtype=bool)
    group_end_mask[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    group_end_indices = np.flatnonzero(group_end_mask)

    tp_at_threshold = true_positive[group_end_indices]
    fp_at_threshold = false_positive[group_end_indices]
    accuracy_at_threshold = (tp_at_threshold + (num_negative - fp_at_threshold)) / total_count
    best_index = int(np.argmax(accuracy_at_threshold))
    if float(accuracy_at_threshold[best_index]) >= best_accuracy:
        score_index = group_end_indices[best_index]
        best_accuracy = float(accuracy_at_threshold[best_index])
        best_threshold = float(sorted_scores[score_index])

    return best_accuracy, best_threshold


def save_roc_plot(output_path: Path, fpr: np.ndarray, tpr: np.ndarray, auc_value: float) -> None:
    positive_mask = fpr > 0
    fig = plt.figure()
    plt.plot(fpr[positive_mask], tpr[positive_mask], lw=1, label=f"glint360k (AUC = {auc_value * 100:.4f}%)")
    plt.xlim([1e-6, 0.1])
    plt.ylim([0.3, 1.0])
    plt.grid(linestyle="--", linewidth=1)
    plt.xticks(FAR_TARGETS)
    plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
    plt.xscale("log")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC on Glint360K")
    plt.legend(loc="lower right")
    fig.savefig(output_path)
    plt.close(fig)


def evaluate_glint360k(
    args: argparse.Namespace,
    feature_extractor: Callable[[list[tuple[str, str, int, int]], Path, Path, str, int, bool], tuple[np.ndarray, np.ndarray, int]] = extract_feature_matrix,
) -> dict[str, object]:
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    index_cache = Path(args.index_cache) if args.index_cache else default_index_cache_path(result_dir)
    output_dir = result_dir / args.job
    output_dir.mkdir(parents=True, exist_ok=True)

    index = load_or_build_index(Path(args.data_root), index_cache)
    eligible_ids, gallery_refs, probe_refs = build_reference_lists(index)
    if len(eligible_ids) < 2:
        raise ValueError("Glint360K evaluation requires at least two eligible identities with 2+ images each.")

    use_flip_test = not args.no_flip_test
    gallery_features, gallery_valid, gallery_invalid = feature_extractor(
        gallery_refs,
        Path(args.data_root),
        Path(args.model_prefix),
        args.network,
        args.batch_size,
        use_flip_test,
    )
    probe_features, probe_valid, probe_invalid = feature_extractor(
        probe_refs,
        Path(args.data_root),
        Path(args.model_prefix),
        args.network,
        args.batch_size,
        use_flip_test,
    )

    valid_mask = gallery_valid & probe_valid
    skipped_identities = int((~valid_mask).sum())
    if skipped_identities:
        print(f"[eval] skipped {skipped_identities} identities because gallery/probe decoding failed")

    filtered_ids = [identity for identity, keep in zip(eligible_ids, valid_mask) if keep]
    gallery_features = gallery_features[valid_mask]
    probe_features = probe_features[valid_mask]
    if len(filtered_ids) < 2:
        raise ValueError("Fewer than two identities remained after filtering invalid gallery/probe images.")

    scores, labels = compute_scores(
        gallery_features,
        probe_features,
        args.num_negatives_per_id,
        args.chunk_size,
    )
    tar_by_far, fpr, tpr, auc_value = compute_roc_metrics(scores, labels)
    verification_accuracy, verification_threshold = compute_verification_accuracy(scores, labels)

    scores_path = output_dir / "scores.npy"
    labels_path = output_dir / "labels.npy"
    summary_path = output_dir / "summary.json"
    roc_path = output_dir / "roc.pdf"
    np.save(scores_path, scores)
    np.save(labels_path, labels)
    save_roc_plot(roc_path, fpr, tpr, auc_value)

    summary = {
        "data_root": str(Path(args.data_root).resolve()),
        "model_prefix": str(Path(args.model_prefix).resolve()),
        "network": args.network,
        "batch_size": args.batch_size,
        "job": args.job,
        "seed": args.seed,
        "use_flip_test": use_flip_test,
        "index_cache": str(index_cache.resolve()),
        "num_eligible_identities": len(eligible_ids),
        "num_evaluated_identities": len(filtered_ids),
        "skipped_identities": skipped_identities,
        "invalid_gallery_images": gallery_invalid,
        "invalid_probe_images": probe_invalid,
        "num_positive_pairs": int(labels.sum()),
        "num_negative_pairs": int((labels == 0).sum()),
        "auc": auc_value,
        "tar_at_far": tar_by_far,
        "verification_accuracy": verification_accuracy,
        "verification_accuracy_threshold": verification_threshold,
        "actual_num_negatives_per_id": int(min(args.num_negatives_per_id, len(filtered_ids) - 1)),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"[eval] eligible identities: {len(eligible_ids)}")
    print(f"[eval] evaluated identities: {len(filtered_ids)}")
    print(f"[eval] positive pairs: {summary['num_positive_pairs']}")
    print(f"[eval] negative pairs: {summary['num_negative_pairs']}")
    print(f"[eval] AUC: {auc_value * 100:.4f}%")
    print(
        f"[eval] Verification Accuracy: {verification_accuracy * 100:.4f}% "
        f"(threshold={verification_threshold:.6f})"
    )
    for far in FAR_TARGETS:
        print(f"[eval] TAR@FAR={far:.0e}: {tar_by_far[str(far)] * 100:.4f}%")
    return summary


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    evaluate_glint360k(args)


if __name__ == "__main__":
    main()
