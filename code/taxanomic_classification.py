#!/usr/bin/env python3
"""
================= Taxonomic Classification of 16S rRNA via Kraken2 + Bracken =================
Taxonomic Classification of 16S rRNA Amplicons
Using Kraken2 + Bracken (SILVA 138 database)
"""

# ================= Taxonoic Classification of 16S rRNA via Kraken2 + Bracken =================
import os
import subprocess
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import logging
import gzip
import psutil
import time
import sys
import re
import shutil

# ==================== EARLY LOGGER SETUP ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---- PATH PLACEHOLDER ----
LOG_FILE = "/path/to/working_directory/process_log.txt"
fh = logging.FileHandler(LOG_FILE)
fh.setLevel(logging.INFO)
logger.addHandler(fh)

# ==================== SYSTEM DEPENDENCIES ====================
logger.info("Installing system dependencies...")
os.system('apt-get update -y')
os.system('apt-get install -y vsearch kraken2 fastp python3-pip build-essential g++ cmake wget tar htop')

# ==================== PYTHON DEPENDENCIES ====================
logger.info("Installing Python dependencies...")
os.system('pip install pandas numpy tqdm psutil')

# ==================== BRACKEN INSTALLATION ====================
bracken_version = "2.9"

# ---- PATH PLACEHOLDERS ----
bracken_dir = f"/tmp/Bracken-{bracken_version}"
bracken_install_dir = "/usr/local/bracken"
bracken_src_dir = f"{bracken_install_dir}/src"

# Clean
logger.info("Cleaning previous Bracken installations...")
os.system(f'rm -rf {bracken_dir} {bracken_install_dir} /tmp/bracken.tar.gz')

# Download & compile
logger.info(f"Downloading and compiling Bracken v{bracken_version}...")
os.system(f'wget -q https://github.com/jenniferlu717/Bracken/archive/refs/tags/v{bracken_version}.tar.gz -O /tmp/bracken.tar.gz')
os.system(f'tar -xzf /tmp/bracken.tar.gz -C /tmp')
os.system(f'cd {bracken_dir}/src && make')

# Install - with proper directory structure
logger.info("Installing Bracken with correct directory structure...")
os.makedirs(bracken_src_dir, exist_ok=True)
os.system(f'cp {bracken_dir}/bracken {bracken_install_dir}/')
os.system(f'cp {bracken_dir}/bracken-build {bracken_install_dir}/')
os.system(f'cp {bracken_dir}/src/est_abundance.py {bracken_src_dir}/')
os.system(f'cp {bracken_dir}/src/kmer2read_distr {bracken_install_dir}/')
os.system(f'chmod +x {bracken_install_dir}/*')
os.system(f'chmod +x {bracken_src_dir}/*')

# Verify installation
logger.info("Verifying Bracken installation...")
for f in ["bracken", "kmer2read_distr"]:
    if not os.path.exists(f"{bracken_install_dir}/{f}"):
        raise FileNotFoundError(f"Bracken file missing: {f}")
if not os.path.exists(f"{bracken_src_dir}/est_abundance.py"):
    raise FileNotFoundError("Bracken est_abundance.py missing from src directory")
logger.info("Bracken installation verified successfully.")

# ==================== KRAKEN2 DB SETUP ====================
# ---- PATH PLACEHOLDERS ----
db_dir = "/path/to/kraken2_databases"
db_subdir = "16S_SILVA138_k2db"
db_path = os.path.join(db_dir, db_subdir)
db_tar = f"{db_dir}/16S_Silva138_20200326.tgz"

os.makedirs(db_dir, exist_ok=True)

if not os.path.exists(os.path.join(db_path, "hash.k2d")):
    logger.info("Downloading Kraken2 16S rRNA Silva138 Database...")
    os.system(f'wget -c https://genome-idx.s3.amazonaws.com/kraken/16S_Silva138_20200326.tgz -O {db_tar}')
    logger.info("Extracting Kraken2 16S database...")
    os.system(f'tar -xvzf {db_tar} -C {db_dir}')
    os.remove(db_tar)

# Verify core files
logger.info("Verifying Kraken2 database files...")
for f in ["hash.k2d", "taxo.k2d", "opts.k2d"]:
    if not os.path.exists(os.path.join(db_path, f)):
        raise FileNotFoundError(f"Missing Kraken2 DB file: {f}")

# Verify Bracken database files are present
logger.info("Verifying Bracken database files...")
bracken_files = glob.glob(f"{db_path}/*mers.kmer_distrib")
if len(bracken_files) == 0:
    logger.error("No Bracken kmer distribution files found! The database may not be Bracken-compatible.")
else:
    logger.info(f"Found {len(bracken_files)} Bracken kmer distribution files.")
    available_read_lengths = sorted([int(f.split('database')[1].split('mers')[0]) for f in bracken_files])
    max_bracken_read_length = max(available_read_lengths)
    logger.info(f"Available Bracken read lengths: {available_read_lengths}")

# ==================== HELPER FUNCTIONS ====================
def monitor_resources():
    cpu_percent = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    logger.info(
        f"System Resources - CPU: {cpu_percent}%, "
        f"Memory: {mem.percent}% used ({mem.used/1024**3:.1f}GB/{mem.total/1024**3:.1f}GB), "
        f"Disk: {disk.percent}% used ({disk.used/1024**3:.1f}GB/{disk.total/1024**3:.1f}GB)"
    )

def run_shell(cmd, check=True, timeout=None):
    logger.debug(f"Running command: {cmd}")
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if check and proc.returncode != 0:
            error_msg = f"Command failed: {cmd}\nstderr: {proc.stderr}\nstdout: {proc.stdout}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout} seconds: {cmd}")
        raise

def estimate_read_length(fastq_file, n_reads=100):
    total_len = 0
    count = 0
    opener = gzip.open if fastq_file.endswith('.gz') else open
    mode = 'rt' if fastq_file.endswith('.gz') else 'r'
    try:
        with opener(fastq_file, mode) as f:
            while count < n_reads:
                header = f.readline()
                if not header:
                    break
                seq = f.readline().strip()
                f.readline()
                f.readline()
                if seq:
                    total_len += len(seq)
                    count += 1
        if count == 0:
            logger.warning(f"No valid reads found in {fastq_file}, using default length 150")
            return 150
        return total_len // max(1, count)
    except Exception as e:
        logger.warning(f"Error estimating read length from {fastq_file}: {e}, using default length 150")
        return 150

def log_kraken_report_stats(report_path):
    if not os.path.exists(report_path):
        logger.warning("Report file does not exist.")
        return 0, 0
    total_assigned = 0
    total_direct = 0
    with open(report_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                assigned = int(parts[2])
                direct = int(parts[1])
                total_assigned += assigned
                total_direct += direct
    logger.info(f"Kraken report stats: {total_direct} direct, {total_assigned} assigned reads classified")
    return total_direct, total_assigned

# ==================== SAMPLE PROCESSING FUNCTION ====================
def process_sample(sample_id, fastqs, kraken_db, bracken_db_dir, max_attempts=2):
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            for fq in fastqs:
                if not os.path.exists(fq):
                    raise ValueError(f"FASTQ file does not exist: {fq}")
                if os.path.getsize(fq) == 0:
                    raise ValueError(f"FASTQ file is empty: {fq}")

            monitor_resources()

            est_len = estimate_read_length(fastqs[0])
            est_len_capped = min(est_len, max_bracken_read_length)
            rl = min(available_read_lengths, key=lambda x: abs(x - est_len_capped))
            logger.info(
                f"Sample {sample_id}: estimated read length = {est_len}, "
                f"using Bracken model = {rl} (capped at {max_bracken_read_length})"
            )

            kraken_out = f"/tmp/{sample_id}.kraken"
            report_path = f"/tmp/{sample_id}.report"
            bracken_out = f"/tmp/{sample_id}.bracken"

            paired_flag = "--paired" if len(fastqs) == 2 else ""
            compressed_flag = "--gzip-compressed" if any(fq.endswith('.gz') for fq in fastqs) else ""
            input_files = " ".join([f"'{fq}'" for fq in fastqs])

            cmd_kraken = (
                f"kraken2 --db '{kraken_db}' {paired_flag} {compressed_flag} {input_files} "
                f"--output '{kraken_out}' --report '{report_path}' "
                f"--threads 4 --use-names --report-zero-counts --memory-mapping --confidence 0"
            )

            run_shell(cmd_kraken, timeout=600)

            if not os.path.exists(report_path) or os.path.getsize(report_path) == 0:
                raise RuntimeError(f"Kraken2 failed to generate report for {sample_id}")

            direct, assigned = log_kraken_report_stats(report_path)
            if assigned == 0:
                logger.warning(f"Sample {sample_id}: No reads classified by Kraken2. Skipping Bracken.")
                return {"SampleID": sample_id}

            cmd_bracken = (
                f"'{bracken_install_dir}/bracken' -d '{kraken_db}' -i '{report_path}' "
                f"-o '{bracken_out}' -r {rl} -l G -t 10"
            )

            run_shell(cmd_bracken, timeout=180)

            if not os.path.exists(bracken_out) or os.path.getsize(bracken_out) == 0:
                raise RuntimeError(f"Bracken failed to generate output for {sample_id}")

            df = pd.read_csv(bracken_out, sep="\t", usecols=["name", "new_est_reads"])
            df["name"] = df["name"].str.strip()
            df = df[df["name"] != "unclassified"]
            df = df[df["new_est_reads"] > 0]

            if len(df) == 0:
                row_data = {}
            else:
                row_data = df.set_index("name")["new_est_reads"].to_dict()

            row_data["SampleID"] = sample_id
            logger.info(f"Successfully processed sample {sample_id} on attempt {attempt} ({len(row_data) - 1} genera)")
            return row_data

        except Exception as e:
            logger.error(f"Attempt {attempt} failed for {sample_id}: {str(e)}")
            if attempt >= max_attempts:
                logger.error(f"All attempts failed for sample {sample_id}")
                return {"SampleID": sample_id}
            else:
                logger.info(f"Retrying sample {sample_id} after cleanup")
                time.sleep(5)
        finally:
            for f in [kraken_out, report_path, bracken_out]:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except Exception as e:
                    logger.warning(f"Error removing temp file {f}: {e}")

    return {"SampleID": sample_id}

# ==================== SAMPLE DISCOVERY ====================
def discover_samples():
    samples = defaultdict(list)
    all_fastqs = glob.glob("/path/to/input_fastqs/**/*.fastq*", recursive=True)
    logger.info(f"Found {len(all_fastqs)} FASTQ files total")

    processed = set()
    standard_pattern = re.compile(r'(.+)_R([12])(?:_\d+)?(?:\.fastq(?:\.gz)?)$', re.IGNORECASE)
    id_pattern = re.compile(r'(ID_[A-Z0-9]+)(?:[_-](\d))?(?:\.fastq(?:\.gz)?)$', re.IGNORECASE)
    generic_pair_pattern = re.compile(r'(.+)[_-]([12])(?:\.fastq(?:\.gz)?)$', re.IGNORECASE)

    for fq in all_fastqs:
        basename = os.path.basename(fq)
        match = standard_pattern.search(basename)
        if match:
            sid = match.group(1)
            samples[sid].append(fq)
            processed.add(fq)

    for fq in all_fastqs:
        if fq in processed:
            continue
        basename = os.path.basename(fq)
        sid = None

        m1 = id_pattern.search(basename)
        if m1:
            sid = m1.group(1)
        else:
            m2 = generic_pair_pattern.search(basename)
            if m2:
                sid = m2.group(1)

        if sid:
            samples[sid].append(fq)
            processed.add(fq)
        else:
            sid = os.path.splitext(os.path.splitext(basename)[0])[0]
            samples[sid] = [fq]
            processed.add(fq)
            logger.warning(f"Using filename as sample ID: {sid} from {fq}")

    for sid in samples:
        samples[sid].sort()

    logger.info(f"Found {len(samples)} unique samples")
    return samples

# ==================== MAIN PIPELINE ====================
def main():
    working_dir = "/path/to/working_directory"
    os.makedirs(working_dir, exist_ok=True)

    output_csv = os.path.join(working_dir, "bracken_genus_abundances.csv")
    if os.path.exists(output_csv):
        os.remove(output_csv)

    samples = discover_samples()
    if not samples:
        logger.error("No samples found to process!")
        return

    all_rows = []
    success = 0
    failures = []

    for sid, fastqs in tqdm(samples.items(), desc="Classifying with Kraken2 + Bracken"):
        row_data = process_sample(sid, fastqs, db_path, db_path)
        all_rows.append(row_data)

        if len(row_data) > 1:
            success += 1
        else:
            failures.append(sid)

    all_species = sorted({k for row in all_rows for k in row if k != "SampleID"})
    cols = ["SampleID"] + all_species

    prepared = [{c: row.get(c, 0) for c in cols} for row in all_rows]
    df = pd.DataFrame(prepared)
    df.to_csv(output_csv, index=False)

    logger.info(f"Processing completed: {success}/{len(samples)} samples successful")

    if failures:
        with open(os.path.join(working_dir, "failed_samples.txt"), "w") as f:
            f.write("\n".join(failures))

if __name__ == "__main__":
    try:
        start = time.time()
        main()
        logger.info(f"Total runtime: {(time.time() - start)/60:.2f} minutes")
    except Exception as e:
        logger.exception("Pipeline failed")
        sys.exit(1)
