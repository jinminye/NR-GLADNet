"""Evaluate NR-GLADNet using prepared features and a saved checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from datetime import datetime
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent

# Edit these three paths, or override them with --split, --weights and --features.
# Relative paths are resolved against this script's directory.
SPLIT_PATH = ROOT / "splits/stft_dcase_win25ms_nfft512_hop10ms_1s_len128_remove_dc_peak_dcase_class_random_train_test_70_30_seed44.json"
WEIGHTS_PATH = ROOT / "checkpoints/DcaseClean/glam_fusion_seed43_20260613-212011/last.pth"
FEATURE_DIR = ROOT / "features/stft_dcase_win25ms_nfft512_hop10ms_1s_len128_remove_dc_peak"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [path.resolve(), (ROOT / path).resolve()]
    if path.parts and path.parts[0].lower() == ROOT.name.lower():
        candidates.append((ROOT.parent / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


def label_from_path(path):
    return path.replace("\\", "/").split("/")[0]


def model_config_from_checkpoint(checkpoint):
    state = checkpoint["model_state"]
    args = checkpoint.get("args", {})
    if args.get("model_variant", "lite_glam_v2") != "lite_glam_v2":
        raise ValueError("This script supports the released LiteGLAMFusionNetV2 checkpoints.")
    return {
        "label_dim": int(state["cnn_head.weight"].shape[0]),
        "cnn_width": int(state["cnn_branch.stem.0.weight"].shape[0]),
        "fusion_dim": int(state["fusion_head.2.weight"].shape[0]),
        "embed_dim": int(state["cls_token"].shape[-1]),
        "dropout": float(args.get("dropout", 0.1)),
        "use_dtnr": any(key.startswith("dtnr.conv1.") for key in state),
        "cnn_norm_type": "ibn" if "cnn_branch.stem.1.instance_norm.weight" in state else "bn" if "cnn_branch.stem.1.running_mean" in state else "ln",
        "convnext_norm_type": "ibn" if "cnn_branch.stage3.1.block.1.instance_norm.weight" in state else "bn" if "cnn_branch.stage3.1.block.1.running_mean" in state else "ln",
        "ast_norm_type": "bn" if "blocks.0.norm1.norm.running_mean" in state else "ln",
        "ablate_no_mb_msa": bool(args.get("ablate_no_mb_msa", False)),
        "ablate_no_gam": bool(args.get("ablate_no_gam", False)),
        "ablate_simple_late_fusion": bool(args.get("ablate_simple_late_fusion", False)),
    }


def build_model(checkpoint):
    spec = importlib.util.spec_from_file_location("nr_gladnet", ROOT / "model/nr-gladnet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = model_config_from_checkpoint(checkpoint)
    model = module.LiteGLAMFusionNetV2(**config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.eval(), config


class TestFeatures(Dataset):
    def __init__(self, feature_dir, split):
        self.feature_dir = feature_dir
        self.paths = [path.replace("\\", "/") for path in split["test"]]
        if not self.paths or len(self.paths) != len(set(self.paths)):
            raise ValueError("The test list must be nonempty and contain no duplicate samples.")
        if split.get("num_test", len(self.paths)) != len(self.paths):
            raise ValueError("num_test does not match the length of the test list.")
        # Original training assigned class indices by sorting labels across both lists.
        self.classes = sorted({label_from_path(path) for path in split.get("train", []) + split["test"]})
        self.class_to_idx = {label: index for index, label in enumerate(self.classes)}
        self.files = [(feature_dir / path).resolve() for path in self.paths]
        for path in self.files:
            if not path.is_relative_to(feature_dir) or not path.is_file():
                raise FileNotFoundError(f"Test feature missing or outside feature directory: {path}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        payload = torch.load(self.files[index], map_location="cpu", weights_only=True)
        feature = payload["feat"].float()
        if feature.ndim == 2:
            feature = feature.unsqueeze(0)
        if tuple(feature.shape) != (1, 128, 128) or not bool(torch.isfinite(feature).all()):
            raise ValueError(f"Expected finite feature tensor [1,128,128]: {self.files[index]}")
        return feature, self.class_to_idx[label_from_path(self.paths[index])]


def classification_metrics(labels, predictions, classes):
    ids = list(range(len(classes)))
    macro = precision_recall_fscore_support(labels, predictions, labels=ids, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(labels, predictions, labels=ids, average="weighted", zero_division=0)
    per_class = precision_recall_fscore_support(labels, predictions, labels=ids, average=None, zero_division=0)
    return {
        "acc": float(accuracy_score(labels, predictions) * 100),
        "macro_precision": float(macro[0]), "macro_recall": float(macro[1]), "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]), "weighted_recall": float(weighted[1]), "weighted_f1": float(weighted[2]),
        "per_class": {name: {"precision": float(per_class[0][i]), "recall": float(per_class[1][i]), "f1": float(per_class[2][i]), "support": int(per_class[3][i])} for i, name in enumerate(classes)},
        "confusion_matrix": confusion_matrix(labels, predictions, labels=ids).tolist(),
    }


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if path.exists():
        backup = path.with_name(path.name + ".before_test_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".bak")
        shutil.copy2(path, backup)
    try:
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if read_json(path) != report:
            raise RuntimeError("Saved report validation failed.")
    except Exception:
        if backup:
            shutil.copy2(backup, path)
        raise


def main():
    parser = argparse.ArgumentParser(description="Test NR-GLADNet using a split, checkpoint and prepared features.")
    parser.add_argument("--split", type=Path, default=SPLIT_PATH)
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--features", type=Path, default=FEATURE_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON result path.")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    split_path, weights, features = map(resolve_path, (args.split, args.weights, args.features))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(43)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    split = read_json(split_path)
    dataset = TestFeatures(features, split)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    model, config = build_model(checkpoint)
    if config["label_dim"] != len(dataset.classes):
        raise ValueError("The weight classifier size does not match the split's classes.")
    print(f"Device: {device}; test samples: {len(dataset)}; classes: {dataset.classes}", flush=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = model.to(device)
    labels, predictions = [], []
    with torch.inference_mode():
        for batch_index, (feature, target) in enumerate(loader, 1):
            logits = model(feature.to(device))
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("The model produced nonfinite outputs.")
            labels.extend(target.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            if batch_index % 10 == 0 or batch_index == len(loader):
                print(f"Evaluated {len(labels)}/{len(dataset)}", flush=True)
    metrics = classification_metrics(labels, predictions, dataset.classes)
    report = {
        "evaluation_time": datetime.now().astimezone().isoformat(),
        "split_path": str(split_path), "weights_path": str(weights), "feature_dir": str(features),
        "split_mode": split.get("split_mode"), "split_seed": split.get("seed"),
        "checkpoint_epoch": checkpoint.get("epoch"), "classes": dataset.classes,
        "num_test": len(labels), "device": str(device), "batch_size": args.batch_size,
        "torch_version": str(torch.__version__), "model_config": config,
        "test": {"fusion": metrics},
    }
    output = resolve_path(args.output) if args.output else ROOT / "results" / (datetime.now().strftime("test_%Y%m%d_%H%M%S_%f") + ".json")
    save_report(output, report)
    print(f"Accuracy: {metrics['acc']:.4f}%")
    print(f"Macro Precision: {100 * metrics['macro_precision']:.4f}%; Recall: {100 * metrics['macro_recall']:.4f}%; F1: {100 * metrics['macro_f1']:.4f}%")
    print("Confusion matrix (rows=true, columns=predicted):")
    for row in metrics["confusion_matrix"]:
        print(row)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
