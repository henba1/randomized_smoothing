"""
Utility script to process incomplete experiment outputs from timeout/interruption.
Current issue: when experiment is terminated by scheduler, we do not end gracefully 
in certify, meaning that the (filtered) csv file is not created).

This script allows reprocessing of .txt output files that were interrupted before
the create_verona_csv step could run. It can optionally filter results and create
VERONA CSV files.

Usage:
    python process_incomplete_results.py --results_dir /path/to/results --filter
    python process_incomplete_results.py --results_dir /path/to/results
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import rs_rd_research_code
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from report_creator import create_filtered_report, create_verona_csv


def extract_params_from_filename(filename: str) -> dict | None:
    """Extract experiment parameters from filename.

    Expected format: {classifier}_{sigma}_{dataset}_{timestamp}.txt
    or similar patterns based on certify.py line 66.

    Args:
        filename: The output filename

    Returns:
        Dictionary with extracted parameters or None if unable to parse
    """
    # Remove extension
    name_without_ext = filename.replace(".txt", "").replace("_filtered", "")
    parts = name_without_ext.split("_")

    # Try to extract key information
    # Assuming format: classifier_sigma_dataset_timestamp
    if len(parts) >= 3:
        try:
            sigma = float(parts[1])
            return {"sigma": sigma}
        except (ValueError, IndexError):
            pass

    return None


def process_result_file(
    result_file: str,
    filter_enabled: bool = False,
    classifier_name: str = "unknown",
    ddpm_model_name: str = "ddpm",
    sigma: float = 1.0,
    alpha: float = 0.001,
    N0: int = 100,
    N: int = 100000,
) -> bool:
    """Process a single result file: filter (optional) and create VERONA CSV.

    Args:
        result_file: Path to the .txt output file
        filter_enabled: Whether to filter misclassified instances
        classifier_name: Name of the classifier model
        ddpm_model_name: Name of the DDPM model
        sigma: Noise level parameter
        alpha: Failure probability parameter
        N0: Number of samples for initial prediction
        N: Number of samples for certification

    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(result_file):
        print(f"Error: File not found: {result_file}")
        return False

    if not result_file.endswith(".txt"):
        print(f"Skipping non-.txt file: {result_file}")
        return False

    # Skip if already filtered or is a CSV
    if "_filtered" in result_file or result_file.endswith(".csv"):
        print(f"Skipping already processed file: {result_file}")
        return False

    print(f"\nProcessing: {result_file}")
    print(f"  Filter enabled: {filter_enabled}")

    # Step 1: Filter if enabled
    input_for_verona = result_file
    if filter_enabled:
        filtered_file = result_file.replace(".txt", "_filtered.txt")
        try:
            create_filtered_report(result_file, filtered_file, experiment=None)
            input_for_verona = filtered_file
            print(f"  ✓ Filtered report created")
        except Exception as e:
            print(f"  ✗ Error creating filtered report: {e}")
            return False

    # Step 2: Create VERONA CSV
    verona_csv_file = input_for_verona.replace(".txt", "_result_df.csv")
    try:
        create_verona_csv(
            input_for_verona,
            verona_csv_file,
            classifier_name,
            ddpm_model_name,
            sigma,
            alpha,
            N0,
            N,
            experiment=None,
        )
        print(f"  ✓ VERONA CSV created")
        return True
    except Exception as e:
        print(f"  ✗ Error creating VERONA CSV: {e}")
        return False


def main(args):
    """Main function to process incomplete results directory."""
    results_dir = args.results_dir

    if not os.path.isdir(results_dir):
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)

    # Find all .txt files in the directory
    txt_files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".txt") and "_filtered" not in f
    ]

    if not txt_files:
        print(f"No unprocessed .txt files found in {results_dir}")
        sys.exit(0)

    print(f"Found {len(txt_files)} .txt file(s) to process")
    print(f"Filter enabled: {args.filter}")
    print(f"Parameters: sigma={args.sigma}, alpha={args.alpha}, N0={args.N0}, N={args.N}")

    successful = 0
    failed = 0

    for txt_file in sorted(txt_files):
        success = process_result_file(
            txt_file,
            filter_enabled=args.filter,
            classifier_name=args.classifier_name,
            ddpm_model_name=args.ddpm_model_name,
            sigma=args.sigma,
            alpha=args.alpha,
            N0=args.N0,
            N=args.N,
        )
        if success:
            successful += 1
        else:
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Processing complete: {successful} successful, {failed} failed")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process incomplete experiment outputs from timeout/interruption"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing unfinished .txt output files",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        default=False,
        help="Filter misclassified instances (prediction == -2) before creating VERONA CSV",
    )
    parser.add_argument(
        "--classifier_name",
        type=str,
        default="unknown",
        help="Name of the classifier model (for VERONA CSV metadata)",
    )
    parser.add_argument(
        "--ddpm_model_name",
        type=str,
        default="ddpm",
        help="Name of the DDPM model (for VERONA CSV metadata)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Noise level parameter (for VERONA CSV metadata)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.001,
        help="Failure probability parameter (for VERONA CSV metadata)",
    )
    parser.add_argument(
        "--N0",
        type=int,
        default=100,
        help="Number of samples for initial prediction (for VERONA CSV metadata)",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=100000,
        help="Number of samples for certification (for VERONA CSV metadata)",
    )

    args = parser.parse_args()
    main(args)

