"""
run_compression_benchmarks.py
=============================
Main Benchmark Runner for Advanced Data Compression.

Executes all compression techniques:
1. Lossless: Canonical Huffman Coding, Gzip/Deflate, LZMA, Bzip2, RLE
2. Lossy Transform: Karhunen-Loève Transform (KLT/PCA) with Quantization
3. Lossy Block Coding: Vector Quantization (VQ / LBG)
4. Deep Representation: Neural Autoencoder Latent Compression
5. Information-Theoretic MDL Anomaly Detection

Outputs:
- Formatted console table
- results/compression/compression_metrics.json
- results/compression/rate_distortion_curve.png
- results/compression/compression_benchmarks.png
- results/compression/mdl_anomaly_distribution.png
"""

import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.compression.evaluator import CompressionBenchmarkEvaluator
from src.preprocessing import build_preprocessor, load_dataset


def main():
    print("=" * 80)
    print(" ADVANCED DATA COMPRESSION: BENCHMARK & EVALUATION PIPELINE")
    print("=" * 80)

    project_root = Path(__file__).resolve().parent
    results_dir = project_root / "results" / "compression"
    models_dir = project_root / "notebooks" / "models"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_path = project_root / "data" / "raw" / "UNSW_NB15_training-set.csv"
    test_path = project_root / "data" / "raw" / "UNSW_NB15_testing-set.csv"
    preprocessor_path = models_dir / "preprocessor.pkl"

    print(f"\n[1/5] Ingesting dataset from {test_path.name}...")
    X_train_df, y_train, _ = load_dataset(train_path)
    X_test_df, y_test, _ = load_dataset(test_path)

    # Load or fit preprocessor
    if preprocessor_path.exists():
        print(f"      Loading preprocessor from {preprocessor_path.name}...")
        preprocessor = joblib.load(preprocessor_path)
        X_train_trans = preprocessor.transform(X_train_df)
        X_test_trans = preprocessor.transform(X_test_df)
    else:
        print("      Fitting ColumnTransformer preprocessor...")
        preprocessor = build_preprocessor()
        X_train_trans = preprocessor.fit_transform(X_train_df)
        X_test_trans = preprocessor.transform(X_test_df)
        joblib.dump(preprocessor, preprocessor_path)

    print(f"      Feature Matrix: {X_test_trans.shape[0]} samples x {X_test_trans.shape[1]} dimensions.")

    # 2. Run Comprehensive Benchmarks
    print("\n[2/5] Running benchmark suite across Lossless & Lossy algorithms...")
    evaluator = CompressionBenchmarkEvaluator(model_dir=models_dir)
    results = evaluator.run_full_benchmark(
        X_train_trans=X_train_trans,
        X_test_trans=X_test_trans,
        y_test=y_test,
        y_train=y_train,
        sample_size=3000
    )

    # 3. Format Console Output
    print("\n[3/5] Benchmark Results Table:")
    print("-" * 125)
    print(f"{'Category':<32} | {'Technique / Algorithm':<35} | {'Type':<8} | {'Decrease %':<10} | {'Comp Ratio':<10} | {'PSNR (dB)':<9} | {'Down F1':<8}")
    print("-" * 125)

    for item in results["benchmarks"]:
        cat = item.get("category", "")[:30]
        algo = item["algorithm"][:33]
        atype = item["type"]
        dec_pct = f"{item['decrease_percentage']:.2f}%"
        cr = f"{item['compression_ratio']:.2f}x"
        psnr = "inf" if item.get("psnr_db") == float("inf") or item.get("psnr_db", 0) >= 999 else f"{item['psnr_db']:.1f}"
        down_f1 = f"{item.get('downstream_f1', 0):.4f}"
        print(f"{cat:<32} | {algo:<35} | {atype:<8} | {dec_pct:<10} | {cr:<10} | {psnr:<9} | {down_f1:<8}")

    print("-" * 125)
    if "vq_vs_sq_advantage_analysis" in results:
        adv = results["vq_vs_sq_advantage_analysis"]
        print("\n[*] Advantages of Vector Quantization over Scalar Quantization:")
        print(f"    - Uniform Scalar Quantization (8-bit) Space Decrease: {adv['scalar_quantizer_8bit_decrease_pct']}%")
        print(f"    - LBG Vector Quantization (K=256) Space Decrease:      {adv['vector_quantizer_k256_decrease_pct']}%")
        print(f"    - Additional Space Saved by VQ:                       +{adv['additional_space_saved_by_vq_pct']}%")
        print(f"    - VQ Advantage Compression Factor:                    {adv['vq_compression_advantage_ratio']}x higher reduction!")

    # 4. Save JSON Metrics
    print(f"\n[4/5] Serializing results to {results_dir / 'compression_metrics.json'}...")
    with open(results_dir / "compression_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 5. Generate Visual Charts
    print(f"\n[5/5] Generating publication-quality visualization figures in {results_dir}...")
    df_res = pd.DataFrame(results["benchmarks"])

    # Figure 1: Compression Ratio & Decrease Percentage
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=df_res,
        x="algorithm",
        y="decrease_percentage",
        palette="crest",
        hue="type"
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.ylabel("Decrease Percentage (%)", fontsize=11, fontweight="bold")
    plt.title("Advanced Data Compression: Space Decrease (%) Across Algorithms", fontsize=13, fontweight="bold")
    plt.ylim(0, 105)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}%",
                        (p.get_x() + p.get_width() / 2., height + 1),
                        ha='center', va='bottom', fontsize=8, rotation=0)
    plt.tight_layout()
    plt.savefig(results_dir / "decrease_percentage_comparison.png", dpi=300)
    plt.close()

    # Figure 2: Rate-Distortion & Utility Curve (Decrease % vs PSNR vs F1)
    lossy_df = df_res[df_res["type"] == "Lossy"].copy()
    if not lossy_df.empty:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        sns.lineplot(
            data=lossy_df,
            x="decrease_percentage",
            y="psnr_db",
            marker="o",
            color="royalblue",
            label="PSNR (dB) - Reconstruction Quality",
            ax=ax1,
            linewidth=2.5
        )
        ax1.set_xlabel("Decrease Percentage (%) [Compression Rate]", fontsize=11, fontweight="bold")
        ax1.set_ylabel("PSNR (dB)", color="royalblue", fontsize=11, fontweight="bold")
        ax1.tick_params(axis="y", labelcolor="royalblue")
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2 = ax1.twinx()
        sns.lineplot(
            data=lossy_df,
            x="decrease_percentage",
            y="f1_retention_pct",
            marker="s",
            color="crimson",
            label="Downstream F1 Retention (%)",
            ax=ax2,
            linewidth=2.5,
            linestyle="--"
        )
        ax2.set_ylabel("Downstream F1 Retention (%)", color="crimson", fontsize=11, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor="crimson")
        plt.title("Rate-Distortion & Task-Utility Trade-off Curve", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(results_dir / "rate_distortion_curve.png", dpi=300)
        plt.close()

    print("\n Benchmark suite execution complete! All artifacts generated successfully.")


if __name__ == "__main__":
    main()
