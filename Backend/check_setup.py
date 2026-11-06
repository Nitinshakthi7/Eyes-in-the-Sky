"""check_setup.py - laptop and environment verification script.

Owner: Nitin (Part D)

Why we use this file:
Setting up deep learning environments across different student laptops often leads to subtle bugs
(wrong Python versions, CPU-only PyTorch instead of CUDA GPU, missing libraries, or outdated drivers).
This script runs a comprehensive pre-flight check on the student's laptop to confirm everything
is ready for training and reproducing project experiments.

What it checks, in order:
  1. Python version (ensures everyone uses 3.12)
  2. Required packages (verifies all libraries from requirements.txt are installed)
  3. Git user configuration (checks that commits are tied to the member's GitHub email)
  4. NVIDIA GPU and driver version (via nvidia-smi)
  5. PyTorch CUDA capability (ensures PyTorch can actually run computations on the GPU)
  6. Free disk space (ensures enough storage for the EuroSAT dataset and model checkpoints)
  7. Performance benchmark speed test (computes estimated time per epoch for ResNet-18)
  8. Optional: automated download of EuroSAT dataset and pretrained weights (--download flag)

Usage:
  python check_setup.py --name Nitin
  python check_setup.py --name Nitin --download
"""

import argparse
import importlib.metadata
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Global list of test outcomes: each item is a tuple (check_name, status, detail_message, recommended_fix)
RESULTS = []

# Dictionary storing key summary stats (Python version, GPU specs, epoch speed, etc.)
INFO = {}

# Required dependencies that every team member must have installed in their virtual environment
PACKAGES = [
    "torch", "torchvision", "numpy", "pandas", "matplotlib", "scikit-learn",
    "pillow", "fastapi", "uvicorn", "python-multipart", "notebook", "ipykernel",
]

# Total number of training images in our 70% split (70% of 27,000 EuroSAT images)
TRAIN_IMAGES = 18900
BATCH = 64


def record(check, status, detail, fix=""):
    """Store one check result in our log and print a clean formatted line to the terminal.

    status: 'OK' (all good), 'WARN' (acceptable but not ideal), or 'FAIL' (blocks training).
    """
    RESULTS.append((check, status, detail, fix))
    line = f"  [{status}] {check}: {detail}"
    if fix and status != "OK":
        line += f"\n         fix: {fix}"
    print(line)


def run_check(title, func):
    """Run an individual check function inside a safety try/except wrapper.

    Why?
    If one check crashes unexpectedly (e.g. nvidia-smi command not found),
    we still want the rest of the checklist to run and give a full report.
    """
    print(f"\n== {title}")
    try:
        func()
    except Exception as err:
        record(title, "FAIL", f"unexpected error: {type(err).__name__}: {err}",
               "send this line to Nitin")


def run_cmd(args, timeout=10):
    """Execute a system shell command safely and return its standard output string.

    Returns an empty string if the command does not exist or fails.
    """
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


# ---------- Individual System Checks ----------

def check_python():
    """Verify that Python 3.12 is being used.

    Why?
    Python 3.12 is the agreed team standard. Different Python versions can have
    incompatible dependency wheels or slightly different random seeding behavior.
    """
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    INFO["python"] = ver
    if (v.major, v.minor) == (3, 12):
        record("Python", "OK", ver)
    elif (v.major, v.minor) in [(3, 10), (3, 11), (3, 13)]:
        record("Python", "WARN", f"{ver} - team standard is 3.12",
               "install Python 3.12 from python.org and make a new .venv with it")
    else:
        record("Python", "FAIL", f"{ver} is not supported",
               "install Python 3.12 from python.org and make a new .venv with it")


def check_packages():
    """Verify that all required Python packages are installed in the active environment."""
    for name in PACKAGES:
        try:
            record(name, "OK", importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            if name in ("torch", "torchvision"):
                fix = "install PyTorch with the command in README, Setup step 4"
            else:
                fix = "pip install -r requirements.txt"
            record(name, "FAIL", "MISSING", fix)


def check_git():
    """Check that Git is installed and that user name and email are configured.

    Why?
    If git user.email is missing or doesn't match the student's GitHub email,
    their commits won't be credited to their profile on the project repository.
    """
    version = run_cmd(["git", "--version"])
    if not version:
        record("Git", "FAIL", "git not found", "install Git from git-scm.com, then reopen PowerShell")
        return
    record("Git", "OK", version)

    name = run_cmd(["git", "config", "user.name"])
    email = run_cmd(["git", "config", "user.email"])
    record("Git user.name", "OK" if name else "WARN", name or "not set",
           'git config --global user.name "Your Name"')
    if email:
        record("Git user.email", "OK", email)
    else:
        record("Git user.email", "WARN", "not set",
               'git config --global user.email "you@example.com"')
    print("  REMINDER: this must be the same email as your GitHub account,")
    print("            or your commits will not count as yours.")


def check_nvidia():
    """Query the operating system for an NVIDIA graphics card using nvidia-smi."""
    out = ""
    # Check default PATH, and also standard Windows installation directory as fallback
    for exe in ["nvidia-smi", r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"]:
        out = run_cmd([exe, "--query-gpu=name,driver_version,memory.total",
                       "--format=csv,noheader"])
        if out:
            break
    if not out:
        record("NVIDIA GPU", "WARN", "no NVIDIA GPU or driver found",
               "the project still runs on CPU, but training will be slow")
        return
    INFO["gpu_line"] = out.splitlines()[0]
    record("NVIDIA GPU", "OK", INFO["gpu_line"])


def check_torch():
    """Verify PyTorch is installed with GPU (CUDA) support rather than CPU-only.

    Why?
    Training on CPU takes ~10 minutes per epoch, while GPU takes ~1-2 seconds per epoch.
    A common student mistake on Windows is running 'pip install torch' which defaults to CPU.
    """
    try:
        import torch
    except ImportError:
        record("PyTorch", "FAIL", "torch is not installed",
               "install PyTorch with the command in README, Setup step 4")
        return

    cuda_build = torch.version.cuda or "none (CPU-only build)"
    cuda_ok = torch.cuda.is_available()
    INFO["torch"] = f"{torch.__version__} + CUDA {cuda_build}"
    record("PyTorch", "OK", f"version {torch.__version__}, CUDA build {cuda_build}, "
                            f"cuda available = {cuda_ok}")

    if cuda_ok:
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        INFO["gpu"] = f"{name}, {vram:.1f} GB VRAM"
        record("PyTorch GPU", "OK", INFO["gpu"])
    elif "gpu_line" in INFO:
        # Hardware has an NVIDIA card, but PyTorch was installed without CUDA support
        INFO["gpu"] = "CPU only (CPU-only PyTorch installed)"
        record("PyTorch GPU", "FAIL", "nvidia-smi found a GPU but PyTorch cannot use it",
               "you installed the CPU-only PyTorch; reinstall using the CUDA command from "
               "https://pytorch.org/get-started/locally/ (Windows, Pip, CUDA)")
    else:
        INFO["gpu"] = "CPU only"
        record("PyTorch GPU", "WARN", "no GPU for PyTorch, will use CPU",
               "fine for this check; training needs a GPU")


def check_disk():
    """Verify there is at least 5 GB of free hard drive space for data and models."""
    free_gb = shutil.disk_usage(ROOT).free / 1024 ** 3
    if free_gb < 5:
        record("Disk space", "WARN", f"only {free_gb:.1f} GB free",
               "free some space: the dataset and PyTorch need several GB")
    else:
        record("Disk space", "OK", f"{free_gb:.1f} GB free")


def make_small_cnn(torch):
    """Helper that creates a quick lightweight 3-block CNN for the speed test."""
    nn = torch.nn

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                             nn.ReLU(), nn.MaxPool2d(2))

    return nn.Sequential(block(3, 32), block(32, 64), block(64, 128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10))


def sync(torch, device):
    """Wait for all asynchronous GPU operations to finish before taking a clock reading."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark(torch, model, device, steps, warmup=5):
    """Simulate a short training loop on synthetic random data and measure images processed per second."""
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    # Create synthetic batch on the selected device
    x = torch.rand(BATCH, 3, 64, 64, device=device)
    y = torch.randint(0, 10, (BATCH,), device=device)

    def one_step():
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

    # Run warmup iterations to allow GPU kernels and caches to initialize
    for _ in range(warmup):
        one_step()
    sync(torch, device)

    # Timed benchmark iterations
    start = time.perf_counter()
    for _ in range(steps):
        one_step()
    sync(torch, device)

    return steps * BATCH / (time.perf_counter() - start)


def check_speed():
    """Benchmark the laptop's compute hardware to estimate how long real training runs will take."""
    try:
        import torch
        import torchvision
    except ImportError:
        record("Speed test", "FAIL", "torch or torchvision missing, cannot test",
               "install PyTorch first (README, Setup step 4)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = 30 if device.type == "cuda" else 5
    print(f"  running on {device}: {steps} timed steps per model, batch {BATCH}, random data")

    models = {
        "Small CNN": lambda: make_small_cnn(torch),
        "ResNet-18": lambda: torchvision.models.resnet18(weights=None, num_classes=10),
    }

    for name, build in models.items():
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            ips = benchmark(torch, build(), device, steps)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            record(f"Speed {name}", "WARN", f"batch {BATCH} does not fit, use 32",
                   "set the batch size to 32 in train.py")
            if name == "ResNet-18":
                INFO["resnet_sec"] = "out of memory"
                verdict(float("inf"))
            continue

        sec_epoch = TRAIN_IMAGES / ips
        detail = (f"{ips:.0f} images/s, {sec_epoch:.0f} s per epoch, "
                  f"{sec_epoch * 20 / 60:.1f} min for 20 epochs")

        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            detail += f", peak VRAM {peak:.2f} GB"

        record(f"Speed {name}", "OK", detail)

        # Use ResNet-18 speed as the benchmark baseline for team role recommendation
        if name == "ResNet-18":
            INFO["resnet_sec"] = f"{sec_epoch:.0f}"
            verdict(sec_epoch)

    print("  NOTE: these are lower bounds, because real runs also load and augment images.")


def verdict(sec_epoch):
    """Provide practical role recommendation for the student based on their hardware speed."""
    if sec_epoch <= 60:
        text = "GPU-ready: can own a training part (B, C or D)"
    elif sec_epoch <= 300:
        text = "OK for small training runs (B)"
    else:
        text = "Take a non-training part (A or E), or train on a teammate's GPU"
    INFO["verdict"] = text
    record("Verdict", "OK", text)


def check_download():
    """Download the official EuroSAT dataset and pretrained ImageNet model weights."""
    from torchvision import datasets, models
    from torchvision.models import MobileNet_V2_Weights, ResNet18_Weights

    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    # Download EuroSAT images (approx 150 MB)
    try:
        ds = datasets.EuroSAT(root=str(data_dir), download=True)
        n, k = len(ds), len(ds.classes)
        ok = n == 27000 and k == 10
        record("EuroSAT dataset", "OK" if ok else "FAIL", f"{n} images, {k} classes",
               "" if ok else "delete the data/eurosat folder and run --download again")
    except Exception as err:
        record("EuroSAT dataset", "FAIL", f"download failed: {err}",
               "check your internet connection and run --download again")

    # Pre-download ImageNet weights for transfer learning models into PyTorch's local cache
    for label, build in [
        ("ResNet-18 weights", lambda: models.resnet18(weights=ResNet18_Weights.DEFAULT)),
        ("MobileNetV2 weights", lambda: models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)),
    ]:
        try:
            build()
            record(label, "OK", "downloaded to the torch cache")
        except Exception as err:
            record(label, "FAIL", f"download failed: {err}",
                   "check your internet connection and run --download again")


# ---------- Report Generation and Summary ----------

def save_summary(name, line):
    """Save the setup report as a Markdown document in docs/setup/<name>.md."""
    folder = ROOT / "docs" / "setup"
    folder.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in name.lower() if c.isalnum())
    rows = ["| Check | Status | Detail |", "|---|---|---|"]
    rows += [f"| {c} | {s} | {d} |" for c, s, d, _ in RESULTS]
    text = (f"# Setup report: {name}\n\n" + "\n".join(rows) +
            f"\n\nTeam line:\n\n    {line}\n")
    (folder / f"{safe}.md").write_text(text, encoding="utf-8")
    return folder / f"{safe}.md"


def main():
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(description="Check this laptop for the EuroSAT project.")
    parser.add_argument("--name", required=True, help="your first name, e.g. Nitin")
    parser.add_argument("--download", action="store_true",
                        help="also download EuroSAT and the pretrained weights")
    args = parser.parse_args()

    # Configure stdout to handle unusual characters gracefully on Windows command lines
    sys.stdout.reconfigure(errors="replace")

    # Run the full test checklist
    run_check("1. Python", check_python)
    run_check("2. Packages", check_packages)
    run_check("3. Git", check_git)
    run_check("4. NVIDIA GPU", check_nvidia)
    run_check("5. PyTorch and GPU", check_torch)
    run_check("6. Disk space", check_disk)
    run_check("7. Speed test (project score)", check_speed)
    if args.download:
        run_check("8. Download data and weights", check_download)

    # Print summary table
    print("\n========== SUMMARY ==========")
    for check, status, detail, _ in RESULTS:
        print(f"{status:5} {check:24} {detail}")

    # Build the single copy-paste line for the group communication channel
    line = (f"{args.name} | Python {INFO.get('python', '?')} | torch {INFO.get('torch', '?')} | "
            f"{INFO.get('gpu', '?')} | ResNet-18 {INFO.get('resnet_sec', '?')} s/epoch | "
            f"{INFO.get('verdict', '?')}")
    print("\nPaste this line in the team chat:")
    print(line)

    path = save_summary(args.name, line)
    print(f"\nSaved to {path}")

    # Exit with error code 1 if any check failed, so CI/CD or setup scripts can detect problems
    failed = any(status == "FAIL" for _, status, _, _ in RESULTS)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
