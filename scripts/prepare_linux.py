#!/usr/bin/env python3
"""Fetch a workspace-local CUDA 13.3 toolset; no sudo or driver changes.

The selected NVIDIA redistributables are enough for these lessons, not a full
CUDA installation. Packages and the libclang wheel are verified against the
SHA-256 values in their publishers' HTTPS metadata.
"""

import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import tarfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(os.environ.get("CUDA_RUST_LESSONS_TOOLS", str(Path.home() / ".cache" / "cuda-rust-lessons" / "tools"))).resolve()
CUDA = TOOLS / "cuda-13.3"
BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
COMPONENTS = (
    "cccl", "cuda_crt", "cuda_cudart", "cuda_nvcc", "cuda_tileiras",
    "libcurand", "libnvvm", "libnvjitlink",
)


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def download(url, destination, expected_hash):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest == expected_hash:
            return
    print(f"Downloading {destination.name}", flush=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise RuntimeError(f"SHA-256 mismatch: {destination.name}")
    temporary.replace(destination)


def unpack_tar(path, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:xz") as archive:
        # Validate both paths and links for Python versions without data filters.
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise RuntimeError(f"Archive path escapes destination: {member.name}")
            if member.issym() or member.islnk():
                base = target.parent if member.issym() else destination
                if not (base / member.linkname).resolve().is_relative_to(destination.resolve()):
                    raise RuntimeError(f"Archive link escapes destination: {member.name}")
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise RuntimeError(f"Unexpected archive entry: {member.name}")
        archive.extractall(destination)


def main():
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise SystemExit("Run this script inside x86_64 Linux / WSL Ubuntu, using python3.")
    manifest = fetch_json(BASE + "redistrib_13.3.0.json")
    for component in COMPONENTS:
        package = manifest[component]["linux-x86_64"]
        path = TOOLS / "downloads" / Path(package["relative_path"]).name
        download(BASE + package["relative_path"], path, package["sha256"])
        extracted = TOOLS / "packages" / component
        stamp = extracted / ".complete"
        if not stamp.exists():
            unpack_tar(path, extracted)
            roots = [p for p in extracted.iterdir() if p.is_dir()]
            if len(roots) != 1:
                raise RuntimeError(f"Unexpected package structure: {component}")
            shutil.copytree(roots[0], CUDA, dirs_exist_ok=True, symlinks=True)
            stamp.write_text(package["sha256"])

    # libclang supplies bindgen's parser. CUDA Rust's code generation uses the
    # LLVM tools in the pinned Rust toolchain, independently of this parser.
    metadata = fetch_json("https://pypi.org/pypi/libclang/18.1.1/json")
    wheel = next(
        p for p in metadata["urls"]
        if "manylinux" in p["filename"] and "x86_64" in p["filename"]
    )
    wheel_path = TOOLS / "downloads" / wheel["filename"]
    download(wheel["url"], wheel_path, wheel["digests"]["sha256"])
    clang_dir = TOOLS / "libclang"
    with zipfile.ZipFile(wheel_path) as archive:
        for name in archive.namelist():
            if not (clang_dir / name).resolve().is_relative_to(clang_dir.resolve()):
                raise RuntimeError("Wheel contains an escaping path")
        archive.extractall(clang_dir)
    # Wheels can put native packages under <distribution>.data/platlib rather
    # than at their final import location. Normalize that installation layout.
    for platlib in clang_dir.glob("*.data/platlib"):
        shutil.copytree(platlib, clang_dir, dirs_exist_ok=True)
    if not (clang_dir / "clang" / "native" / "libclang.so").is_file():
        raise RuntimeError("The libclang wheel did not supply clang/native/libclang.so")

    # NVIDIA binaries retain the executable permissions recorded in the tar.
    # Store bulk files in the Linux filesystem: extracting thousands of CUDA
    # headers onto /mnt/c or /mnt/d is especially slow under WSL.
    environment = ROOT / ".tools" / "env.sh"
    environment.parent.mkdir(parents=True, exist_ok=True)
    environment.write_text(
        '# Source this file from Bash before running a GPU project.\n'
        f'task_tools_dir={shlex.quote(str(TOOLS))}\n'
        'export CUDA_TOOLKIT_PATH="$task_tools_dir/cuda-13.3"\n'
        'export CUDA_HOME="$CUDA_TOOLKIT_PATH"\n'
        'export PATH="$task_tools_dir/../cli/bin:$CUDA_TOOLKIT_PATH/bin:$PATH"\n'
        'export LD_LIBRARY_PATH="$CUDA_TOOLKIT_PATH/lib:$CUDA_TOOLKIT_PATH/lib64:'
        '$CUDA_TOOLKIT_PATH/nvvm/lib64:${LD_LIBRARY_PATH:-}"\n'
        'export LIBCLANG_PATH="$task_tools_dir/libclang/clang/native"\n'
        '# Supply standard C headers when no standalone clang is installed.\n'
        'export BINDGEN_EXTRA_CLANG_ARGS="-isystem $(gcc -print-file-name=include)"\n'
        'export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$task_tools_dir/../target}"\n'
        'unset task_tools_dir\n'
    )
    print(f"Ready. Run: source {environment}", flush=True)
    print("This did not install Rust, cargo-oxide, or an NVIDIA driver.", flush=True)


if __name__ == "__main__":
    main()
