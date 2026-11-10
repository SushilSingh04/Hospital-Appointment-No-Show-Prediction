"""End-to-end reproducible training, calibration, and evaluation pipeline runner."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_generator import save_synthetic_data
from src.eda import run_data_quality_and_eda
from src.feature_engineering import prepare_and_save_splits
from src.train import train_and_evaluate_all
from src.evaluate import generate_evaluation_figures, generate_markdown_report


def run_full_pipeline():
    print("=" * 70)
    print("STEP 1: Generating Synthetic Hospital Appointments Dataset...")
    print("=" * 70)
    save_synthetic_data()

    print("\n" + "=" * 70)
    print("STEP 2: Executing Data Quality & Leakage Analysis...")
    print("=" * 70)
    eda_report = run_data_quality_and_eda()

    print("\n" + "=" * 70)
    print("STEP 3: Chronological Splitting & Feature Engineering...")
    print("=" * 70)
    prepare_and_save_splits()

    print("\n" + "=" * 70)
    print("STEP 4: Training Models, Calibration & Optimal Threshold Search...")
    print("=" * 70)
    metadata = train_and_evaluate_all()

    print("\n" + "=" * 70)
    print("STEP 5: Generating Evaluation Figures and Final Governance Report...")
    print("=" * 70)
    generate_evaluation_figures()
    generate_markdown_report()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Artifacts saved in {Path('artifacts').resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()
