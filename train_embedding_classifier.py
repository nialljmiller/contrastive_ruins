"""Supervised classifier training on encoder embeddings."""
import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from shapely.geometry import Point, box
from shapely.ops import unary_union
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import sample_random_patches, sample_site_patches
from model import load_models
from visualization import compute_embeddings


DEFAULT_RASTERS = [
    "rasters/Tile 1.tif",
    "rasters/Tile 3.tif",
    "rasters/Tile 7.tif",
    "rasters/Tile 13.tif",
    "rasters/Tile 17.tif",
    "rasters/Tile 20.tif",
    "rasters/Tile 4.tif",
    "rasters/Tile 8.tif",
    "rasters/Tile 10.tif",
    "rasters/Tile 14.tif",
    "rasters/Tile 18.tif",
    "rasters/Tile 2.tif",
    "rasters/Tile 5.tif",
    "rasters/Tile 9.tif",
    "rasters/Tile 11.tif",
    "rasters/Tile 15.tif",
    "rasters/Tile 19.tif",
    "rasters/Tile_3_hillshade.tif",
    "rasters/Tile 6.tif",
    "rasters/Tile 12.tif",
    "rasters/Tile 16.tif",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a supervised classifier on encoder embeddings of site patches"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/project/galacticbulge/ruin_repo",
        help="Base directory that contains rasters, models, etc.",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="site_locations/site_locations.csv",
        help="Relative path to the CSV of known sites (within data_path)",
    )
    parser.add_argument(
        "--site_rasters",
        nargs="+",
        default=DEFAULT_RASTERS,
        help="Raster files (relative to data_path) used to extract positive site patches",
    )
    parser.add_argument(
        "--background_rasters",
        nargs="+",
        default=DEFAULT_RASTERS,
        help="Raster files (relative to data_path) used to sample negative/background patches",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Patch size in pixels",
    )
    parser.add_argument(
        "--negatives_per_raster",
        type=int,
        default=5000,
        help="Number of random background patches sampled per raster",
    )
    parser.add_argument(
        "--negative_buffer",
        type=float,
        default=30.0,
        help="Buffer (in CRS units) around known sites to exclude from negatives",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory (relative to data_path) containing the trained encoder",
    )
    parser.add_argument(
        "--embedding_batch",
        type=int,
        default=512,
        help="Batch size for embedding inference",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for validation",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for the train/test split",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Apply feature standardization before the classifier",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="models/embedding_classifier.joblib",
        help="Path (relative to data_path) where the classifier pipeline will be stored",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="reports/embedding_classifier",
        help="Directory (relative to data_path) where diagnostics will be written",
    )
    parser.add_argument(
        "--save_embeddings",
        type=str,
        default=None,
        help="If provided, save the labeled embeddings to this npz file (relative to data_path)",
    )
    return parser.parse_args()


def load_sites_csv(csv_path: str, crs: str = "EPSG:26912") -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path)
    if {"true_easting", "true_northing"}.issubset(df.columns):
        geometry = [Point(x, y) for x, y in zip(df["true_easting"], df["true_northing"])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    elif {"longitude", "latitude"}.issubset(df.columns):
        geometry = [Point(x, y) for x, y in zip(df["longitude"], df["latitude"])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    else:
        raise ValueError("CSV must contain either true_easting/true_northing or longitude/latitude columns")
    return gdf


def reproject_sites(sites_gdf: gpd.GeoDataFrame, raster_path: str) -> Tuple[gpd.GeoDataFrame, str]:
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
    if raster_crs is None:
        raise ValueError(f"Raster {raster_path} has undefined CRS")
    if sites_gdf.crs != raster_crs:
        return sites_gdf.to_crs(raster_crs), raster_crs.to_string()
    return sites_gdf, raster_crs.to_string()


def collect_positive_embeddings(
    encoder,
    raster_paths: Sequence[str],
    sites_gdf: gpd.GeoDataFrame,
    patch_size: int,
    batch_size: int,
) -> np.ndarray:
    embeddings: List[np.ndarray] = []
    for raster_path in raster_paths:
        try:
            aligned_sites, _ = reproject_sites(sites_gdf, raster_path)
            patches, _, _ = sample_site_patches([raster_path], aligned_sites, patch_size=patch_size)
            if len(patches) == 0:
                continue
            emb = compute_embeddings(encoder, patches, batch_size=batch_size)
            if len(emb):
                embeddings.append(emb)
        except Exception as exc:
            print(f"Warning: skipping positive extraction for {raster_path}: {exc}")
    if not embeddings:
        raise RuntimeError("No positive site embeddings were extracted. Check your inputs.")
    return np.vstack(embeddings)


def filter_background_patches(
    patches: np.ndarray,
    locations: Sequence[Tuple[float, float, float, float]],
    buffered_sites,
) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]]]:
    if buffered_sites is None:
        return patches, list(locations)
    keep_indices: List[int] = []
    for idx, loc in enumerate(locations):
        candidate = box(*loc)
        if not candidate.intersects(buffered_sites):
            keep_indices.append(idx)
    if not keep_indices:
        return np.empty((0,) + patches.shape[1:], dtype=patches.dtype), []
    filtered_patches = patches[keep_indices]
    filtered_locations = [locations[i] for i in keep_indices]
    return filtered_patches, filtered_locations


def collect_negative_embeddings(
    encoder,
    raster_paths: Sequence[str],
    sites_gdf: gpd.GeoDataFrame,
    patch_size: int,
    n_per_raster: int,
    buffer_distance: float,
    batch_size: int,
) -> np.ndarray:
    embeddings: List[np.ndarray] = []
    for raster_path in raster_paths:
        try:
            aligned_sites, _ = reproject_sites(sites_gdf, raster_path)
            patches, locations, _ = sample_random_patches(
                [raster_path],
                patch_size=patch_size,
                n_samples=max(n_per_raster, 0),
                save_dir=None,
            )
            if len(patches) == 0:
                continue
            buffered_sites = None
            if buffer_distance > 0 and not aligned_sites.empty:
                buffered_sites = unary_union(aligned_sites.geometry.buffer(buffer_distance))
            filtered_patches, _ = filter_background_patches(patches, locations, buffered_sites)
            if len(filtered_patches) == 0:
                print(f"All background patches near {raster_path} overlapped known sites; skipping")
                continue
            emb = compute_embeddings(encoder, filtered_patches, batch_size=batch_size)
            if len(emb):
                embeddings.append(emb)
        except Exception as exc:
            print(f"Warning: skipping negative extraction for {raster_path}: {exc}")
    if not embeddings:
        raise RuntimeError("No negative/background embeddings were extracted. Adjust sampling parameters.")
    return np.vstack(embeddings)


def build_classifier_pipeline(standardize: bool) -> Pipeline:
    steps = []
    if standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
            ),
        )
    )
    return Pipeline(steps)


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def save_classification_report(report: str, results_dir: str) -> str:
    ensure_dir(results_dir)
    report_path = os.path.join(results_dir, "validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report_path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    results_dir: str,
) -> str:
    ensure_dir(results_dir)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(cm, display_labels=["Background", "Site"])
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Validation Confusion Matrix")
    fig.tight_layout()
    output_path = os.path.join(results_dir, "confusion_matrix.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    results_dir: str,
) -> Tuple[str, float]:
    ensure_dir(results_dir)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Validation ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path = os.path.join(results_dir, "roc_curve.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path, auc


def plot_probability_histogram(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    results_dir: str,
) -> str:
    ensure_dir(results_dir)
    fig, ax = plt.subplots()
    ax.hist(y_scores[y_true == 1], bins=30, alpha=0.6, label="Sites")
    ax.hist(y_scores[y_true == 0], bins=30, alpha=0.6, label="Background")
    ax.set_xlabel("Predicted Probability of Site")
    ax.set_ylabel("Count")
    ax.set_title("Validation Probability Distribution")
    ax.legend()
    fig.tight_layout()
    output_path = os.path.join(results_dir, "probability_histogram.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def write_metrics(metrics: Dict[str, float], results_dir: str) -> str:
    ensure_dir(results_dir)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics_path


def main() -> None:
    args = parse_args()
    data_path = args.data_path
    csv_path = os.path.join(data_path, args.csv_file)
    site_rasters = [os.path.join(data_path, r) for r in args.site_rasters]
    background_rasters = [os.path.join(data_path, r) for r in args.background_rasters]
    output_path = os.path.join(data_path, args.output_path)
    results_dir = os.path.join(data_path, args.results_dir)
    embeddings_path = os.path.join(data_path, args.save_embeddings) if args.save_embeddings else None

    print("Loading encoder…")
    encoder, _ = load_models(save_dir=os.path.join(data_path, args.model_dir))

    print("Loading known site locations…")
    sites_gdf = load_sites_csv(csv_path)

    print("Extracting positive site embeddings…")
    positive_embeddings = collect_positive_embeddings(
        encoder,
        site_rasters,
        sites_gdf,
        args.patch_size,
        args.embedding_batch,
    )
    print(f"Collected {len(positive_embeddings)} positive embeddings")

    print("Extracting background embeddings…")
    negative_embeddings = collect_negative_embeddings(
        encoder,
        background_rasters,
        sites_gdf,
        args.patch_size,
        args.negatives_per_raster,
        args.negative_buffer,
        args.embedding_batch,
    )
    print(f"Collected {len(negative_embeddings)} negative embeddings")

    X = np.vstack([positive_embeddings, negative_embeddings])
    y = np.concatenate([
        np.ones(len(positive_embeddings), dtype=np.int32),
        np.zeros(len(negative_embeddings), dtype=np.int32),
    ])

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    pipeline = build_classifier_pipeline(args.standardize)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    report = classification_report(y_val, y_pred, digits=3)
    print("Validation report:\n" + report)
    report_path = save_classification_report(report, results_dir)
    print(f"Saved classification report to {report_path}")

    classifier = pipeline.named_steps["clf"]
    metrics: Dict[str, float] = {}
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="binary"
    )
    accuracy = accuracy_score(y_val, y_pred)
    metrics.update(
        {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    )
    cm_path = plot_confusion_matrix(y_val, y_pred, results_dir)
    print(f"Saved confusion matrix plot to {cm_path}")

    if hasattr(classifier, "predict_proba"):
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        roc_path, auc = plot_roc_curve(y_val, y_proba, results_dir)
        print(f"Validation ROC AUC: {auc:.3f}")
        print(f"Saved ROC curve to {roc_path}")
        prob_hist_path = plot_probability_histogram(y_val, y_proba, results_dir)
        print(f"Saved probability histogram to {prob_hist_path}")
        metrics["roc_auc"] = auc
    metrics_path = write_metrics(metrics, results_dir)
    print(f"Saved metrics summary to {metrics_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"Saved classifier pipeline to {output_path}")

    if embeddings_path:
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        np.savez_compressed(
            embeddings_path,
            embeddings=X,
            labels=y,
            positive_count=len(positive_embeddings),
            negative_count=len(negative_embeddings),
        )
        print(f"Saved embeddings to {embeddings_path}")


if __name__ == "__main__":
    main()
