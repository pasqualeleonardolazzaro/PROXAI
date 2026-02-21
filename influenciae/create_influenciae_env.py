import sys
import subprocess
import shutil
from pathlib import Path
import platform

# Path to the isolated env
ENV_DIR = Path(".influenciae_env")


def find_python39():
    """
    Returns a command list to invoke Python 3.9, or raises an error if not found.
    Works on Windows, macOS, Linux.
    """

    # --- Windows ---
    if platform.system() == "Windows":
        # Try Windows py launcher first
        py_launcher = shutil.which("py")
        if py_launcher:
            # "py -3.9" guarantees correct interpreter
            return ["py", "-3.9"]

        # Fallback: direct python3.9.exe
        py39 = shutil.which("python3.9") or shutil.which("python39")
        if py39:
            return [py39]

    # --- Linux / macOS ---
    else:
        py39 = shutil.which("python3.9")
        if py39:
            return [py39]

    # Not found on system
    raise RuntimeError(
        "Python 3.9 not found.\n"
        "Please install Python 3.9 so the Influenciae environment can be created.\n"
        "Windows: https://www.python.org/downloads/windows/\n"
        "Ubuntu/Debian: sudo apt install python3.9\n"
        "macOS (brew): brew install python@3.9\n"
    )


def create_env():
    """
    Creates a dedicated Influenciae environment using Python 3.9
    with pinned dependency versions.
    """

    if ENV_DIR.exists():
        print("[INFO] Influenciae environment already exists.")
        return

    print("[INFO] Creating Influenciae environment with Python 3.9...")

    # -----------------------------------------------------
    # 1. Locate Python 3.9
    # -----------------------------------------------------
    python39_cmd = find_python39()

    # -----------------------------------------------------
    # 2. Create venv using Python 3.9 explicitly
    # -----------------------------------------------------
    subprocess.check_call(python39_cmd + ["-m", "venv", str(ENV_DIR)])

    # -----------------------------------------------------
    # 3. Determine python inside venv
    # -----------------------------------------------------
    if platform.system() == "Windows":
        python_path = ENV_DIR / "Scripts" / "python.exe"
    else:
        python_path = ENV_DIR / "bin" / "python"

    if not python_path.exists():
        raise RuntimeError("[ERROR] Venv was created, but python executable inside it is missing.")

    # -----------------------------------------------------
    # 4. Upgrade pip
    # -----------------------------------------------------
    print("[INFO] Upgrading pip...")
    subprocess.check_call([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])

    # -----------------------------------------------------
    # 5. Install dependencies compatible with Python 3.9
    # -----------------------------------------------------
    print("[INFO] Installing Influenciae dependencies...")
    subprocess.check_call([
        str(python_path), "-m", "pip", "install",
        "numpy==1.22",
        "scipy==1.7",
        "networkx==2.6",
        "pandas==1.2",
        "scikit-learn==1.0",
        "influenciae"
    ])

    print("[DONE] Influenciae environment created successfully!")


# Optional helper to get python from this env
def get_influenciae_python():
    return ENV_DIR / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")


if __name__ == "__main__":
    create_env()