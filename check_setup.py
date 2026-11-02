"""check_setup.py - laptop check for the Group 5 EuroSAT project (owner: Nitin, part D).

Usage:
  python check_setup.py --name Tejash              (checks + speed test)
  python check_setup.py --name Tejash --download   (also downloads EuroSAT and pretrained weights)

Checks, in order:
  1. Python version
  2. Installed packages
  3. Git and your git email
  4. NVIDIA GPU (nvidia-smi)
  5. PyTorch and CUDA
  6. Free disk space
  7. Speed test (the "project score")
  8. Optional: download EuroSAT and pretrained weights (--download)

Each check prints OK / WARN / FAIL with a one-line fix.
Writes docs/setup/<name>.md and prints one line for the team chat.
Exit code is 1 if any check FAILed, else 0.
"""

import argparse
import importlib.metadata
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = []  # each entry: (check, status, detail, fix)
INFO = {}     # values used for the summary line

PACKAGES = [
    "torch", "torchvision", "numpy", "pandas", "matplotlib", "scikit-learn",
    "pillow", "fastapi", "uvicorn", "python-multipart", "notebook", "ipykernel",
]
TRAIN_IMAGES = 18900  # 70% of 27,000
BATCH = 64


def record(check, status, detail, fix=""):
    """Store one result and print it."""
    RESULTS.append((check, status, detail, fix))
    line = f"  [{status}] {check}: {detail}"
    if fix and status != "OK":
        line += f"\n         fix: {fix}"
    print(line)


def run_check(title, func):
    """Run one check. If it crashes, record FAIL and keep going."""
    print(f"\n== {title}")
    try:
        func()
    except Exception as err:  # never stop the other checks
        record(title, "FAIL", f"unexpected error: {type(err).__name__}: {err}",
               "send this line to Nitin")


def run_cmd(args, timeout=10):
    """Run a command and return its text output, or '' if it fails."""
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


# ---------- checks ----------

def check_python():
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
    # nvidia-smi is normally on PATH; on some laptops it is only in the NVIDIA folder
    out = ""
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
        INFO["gpu"] = "CPU only (CPU-only PyTorch installed)"
        record("PyTorch GPU", "FAIL", "nvidia-smi found a GPU but PyTorch cannot use it",
               "you installed the CPU-only PyTorch; reinstall using the CUDA command from "
               "https://pytorch.org/get-started/locally/ (Windows, Pip, CUDA)")
    else:
        INFO["gpu"] = "CPU only"
        record("PyTorch GPU", "WARN", "no GPU for PyTorch, will use CPU",
               "fine for this check; training needs a GPU")


def check_disk():
    free_gb = shutil.disk_usage(ROOT).free / 1024 ** 3
    if free_gb < 5:
        record("Disk space", "WARN", f"only {free_gb:.1f} GB free",
               "free some space: the dataset and PyTorch need several GB")
    else:
        record("Disk space", "OK", f"{free_gb:.1f} GB free")


def make_small_cnn(torch):
    """Small scratch CNN: 3 blocks of Conv-BatchNorm-ReLU-MaxPool, then a Linear to 10 classes."""
    nn = torch.nn

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                             nn.ReLU(), nn.MaxPool2d(2))

    return nn.Sequential(block(3, 32), block(32, 64), block(64, 128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10))


def sync(torch, device):
    """GPU runs are asynchronous: wait for them to finish before reading the clock."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark(torch, model, device, steps, warmup=5):
    """Train on random 64x64 data and return images per second."""
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    x = torch.rand(BATCH, 3, 64, 64, device=device)
    y = torch.randint(0, 10, (BATCH,), device=device)

    def one_step():
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

    for _ in range(warmup):
        one_step()
    sync(torch, device)
    start = time.perf_counter()
    for _ in range(steps):
        one_step()
    sync(torch, device)
    return steps * BATCH / (time.perf_counter() - start)


def check_speed():
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
            if name == "ResNet-18":  # no timing possible, so give the cautious verdict
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
        if name == "ResNet-18":
            INFO["resnet_sec"] = f"{sec_epoch:.0f}"
            verdict(sec_epoch)
    print("  NOTE: these are lower bounds, because real runs also load and augment images.")


def verdict(sec_epoch):
    """Turn ResNet-18 seconds per epoch into a plain advice line."""
    if sec_epoch <= 60:
        text = "GPU-ready: can own a training part (B, C or D)"
    elif sec_epoch <= 300:
        text = "OK for small training runs (B)"
    else:
        text = "Take a non-training part (A or E), or train on a teammate's GPU"
    INFO["verdict"] = text
    record("Verdict", "OK", text)


def check_download():
    from torchvision import datasets, models
    from torchvision.models import MobileNet_V2_Weights, ResNet18_Weights

    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    try:
        ds = datasets.EuroSAT(root=str(data_dir), download=True)
        n, k = len(ds), len(ds.classes)
        ok = n == 27000 and k == 10
        record("EuroSAT dataset", "OK" if ok else "FAIL", f"{n} images, {k} classes",
               "" if ok else "delete the data/eurosat folder and run --download again")
    except Exception as err:
        record("EuroSAT dataset", "FAIL", f"download failed: {err}",
               "check your internet connection and run --download again")
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


# ---------- summary ----------

def save_summary(name, line):
    """Write docs/setup/<name>.md, overwriting it if it exists."""
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
    parser = argparse.ArgumentParser(description="Check this laptop for the EuroSAT project.")
    parser.add_argument("--name", required=True, help="your first name, e.g. Tejash")
    parser.add_argument("--download", action="store_true",
                        help="also download EuroSAT and the pretrained weights")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")  # an odd character in an error must not crash the printing

    run_check("1. Python", check_python)
    run_check("2. Packages", check_packages)
    run_check("3. Git", check_git)
    run_check("4. NVIDIA GPU", check_nvidia)
    run_check("5. PyTorch and GPU", check_torch)
    run_check("6. Disk space", check_disk)
    run_check("7. Speed test (project score)", check_speed)
    if args.download:
        run_check("8. Download data and weights", check_download)

    print("\n========== SUMMARY ==========")
    for check, status, detail, _ in RESULTS:
        print(f"{status:5} {check:24} {detail}")

    line = (f"{args.name} | Python {INFO.get('python', '?')} | torch {INFO.get('torch', '?')} | "
            f"{INFO.get('gpu', '?')} | ResNet-18 {INFO.get('resnet_sec', '?')} s/epoch | "
            f"{INFO.get('verdict', '?')}")
    print("\nPaste this line in the team chat:")
    print(line)
    path = save_summary(args.name, line)
    print(f"\nSaved to {path}")

    failed = any(status == "FAIL" for _, status, _, _ in RESULTS)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
