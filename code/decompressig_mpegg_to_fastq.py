#!/usr/bin/env python3
"""
MPEG-G (.mgb) to FASTQ Processing Pipeline

Uses Docker-based Genie decompression to convert .mgb files into FASTQ format
with parallel execution and robust error handling.
"""

import os
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# =========================
# Configuration
# =========================
MAX_WORKERS = 4
BATCH_SIZE = 10


# =========================
# Processing Functions
# =========================
def process_single_file(mgb_file: str, source_dir: str, dest_dir: str):
    """
    Process a single MPEG-G (.mgb) file using Docker-based Genie decompression.

    Returns:
        (success: bool, filename: str, error: Optional[str])
    """
    try:
        base_name = os.path.splitext(mgb_file)[0]
        temp_fastq = os.path.join(source_dir, f"{base_name}.fastq")

        command = [
            "docker", "run", "--rm",
            "--memory=8g", "--memory-swap=12g",
            "-v", f"{source_dir}:/data",
            "muefab/genie:latest", "run",
            "-f",
            "-i", f"/data/{mgb_file}",
            "-o", f"/data/{base_name}.fastq"
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return False, mgb_file, result.stderr.strip()

        if os.path.exists(temp_fastq):
            final_path = os.path.join(dest_dir, f"{base_name}.fastq")
            shutil.move(temp_fastq, final_path)
            return True, mgb_file, None

        return False, mgb_file, "Output FASTQ file not created"

    except Exception as exc:
        return False, mgb_file, str(exc)


def process_all_files(source_dir: str, dest_dir: str, max_workers: int = MAX_WORKERS):
    """
    Process all .mgb files in a directory using parallel execution.
    """
    os.makedirs(dest_dir, exist_ok=True)

    mgb_files = [
        f for f in os.listdir(source_dir)
        if f.lower().endswith(".mgb")
    ]

    results = {
        "success": 0,
        "failed": 0,
        "errors": []
    }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_file, mgb_file, source_dir, dest_dir)
            for mgb_file in mgb_files
        ]

        with tqdm(total=len(mgb_files), desc="Processing Files") as progress:
            for future in futures:
                success, filename, error = future.result()

                if success:
                    results["success"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append((filename, error))

                progress.update(1)

                if results["success"] > 0 and results["success"] % BATCH_SIZE == 0:
                    progress.set_postfix(
                        Success=results["success"],
                        Failed=results["failed"]
                    )

    return results


# =========================
# Main Execution
# =========================
def main():
    train_dir = "path to mpegg files"
    output_train = "processed_data/train"

    print("Starting training files processing...")
    train_results = process_all_files(train_dir, output_train)

    print("\nProcessing Complete!")
    print(
        f"Training Files - "
        f"Success: {train_results['success']}, "
        f"Failed: {train_results['failed']}"
    )

    if train_results["failed"] > 0:
        with open("processing_errors.log", "w") as log_file:
            log_file.write("Training File Errors:\n")
            for filename, error in train_results["errors"]:
                log_file.write(f"{filename}: {error}\n")

        print("Error log saved to processing_errors.log")


if __name__ == "__main__":
    main()
