"""Проверка корректности submission против эталона NumPy.

Запуск из каталога лабораторной после сборки:
    python3 tests/check.py [--binary build/submission]
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
SEED = 2026
MODES = ["coalesced", "noncoalesced"]
SIZES = [1, 2, 31, 32, 33, 255, 256, 257, 1000, 1 << 20]
# Сложение и ReLU должны совпадать точно. SAXPY допускает FMA: одно округление вместо двух.
# Разница FMA и раздельного умножения/сложения ограничена величиной операндов, а не результата
# (при alpha*x ≈ -y результат близок к нулю), поэтому допуск считается от |alpha*x| + |y|.
SAXPY_EPS_FACTOR = 2.0
EPS32 = np.finfo(np.float32).eps


def write_input(path: Path, alpha: np.float32, x: np.ndarray, y: np.ndarray) -> None:
    with path.open("w") as f:
        f.write(f"{len(x)} {alpha:.9g}\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi:.9g} {yi:.9g}\n")


def reference(alpha: np.float32, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    add = x + y
    saxpy = alpha * x + y
    relu = np.maximum(np.float32(0), x)
    return np.stack([add, saxpy, relu], axis=1)


def read_output(path: Path, n: int) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float32, ndmin=2)
    if n == 0:
        return np.empty((0, 3), dtype=np.float32)
    if data.shape != (n, 3):
        raise ValueError(f"unexpected output shape {data.shape}, expected ({n}, 3)")
    return data


def make_case(rng: np.random.Generator, n: int):
    alpha = np.float32(rng.uniform(-4, 4))
    x = rng.uniform(-100, 100, n).astype(np.float32)
    y = rng.uniform(-100, 100, n).astype(np.float32)
    # Нули и значения разных знаков в начале массива.
    special = np.array([0.0, -0.0, 1.0, -1.0, 1e-30, -1e-30], dtype=np.float32)
    k = min(n, len(special))
    x[:k] = special[:k]
    return alpha, x, y


def run_case(binary: Path, workdir: Path, n: int, mode: str, rng: np.random.Generator) -> bool:
    alpha, x, y = make_case(rng, n)
    inp = workdir / f"in_{n}.txt"
    out = workdir / f"out_{n}_{mode}.txt"
    write_input(inp, alpha, x, y)
    proc = subprocess.run(
        [str(binary), "--input", str(inp), "--output", str(out), "--mode", mode],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"FAIL n={n} mode={mode}: exit {proc.returncode}: {proc.stderr.strip()}")
        return False
    got = read_output(out, n)
    exp = reference(alpha, x, y)
    add_ok = np.array_equal(got[:, 0], exp[:, 0])
    relu_ok = np.array_equal(got[:, 2], exp[:, 2])
    saxpy_diff = np.abs(got[:, 1].astype(np.float64) - exp[:, 1].astype(np.float64))
    saxpy_tol = (
        SAXPY_EPS_FACTOR
        * EPS32
        * (np.abs(alpha.astype(np.float64) * x) + np.abs(y.astype(np.float64)))
    )
    saxpy_ok = bool(np.all(saxpy_diff <= saxpy_tol))
    saxpy_err = float(np.max(saxpy_diff)) if n else 0.0
    # Доля допуска: <= 1 — проходит; 0 — точное совпадение с раздельным округлением.
    nonzero = saxpy_tol > 0
    saxpy_ratio = float(np.max(saxpy_diff[nonzero] / saxpy_tol[nonzero])) if nonzero.any() else 0.0
    ok = add_ok and relu_ok and saxpy_ok
    status = "OK  " if ok else "FAIL"
    print(
        f"{status} n={n:<8} mode={mode:<12} add_exact={add_ok} relu_exact={relu_ok} "
        f"saxpy_max_abs_err={saxpy_err:.3e} saxpy_err/tol={saxpy_ratio:.3f}"
    )
    return ok


def run_example(binary: Path, workdir: Path, mode: str) -> bool:
    """Пример из задания: data/example.txt должен дать data/example_expected.txt."""
    out = workdir / f"example_{mode}.txt"
    proc = subprocess.run(
        [str(binary), "--input", str(DATA / "example.txt"), "--output", str(out), "--mode", mode],
        capture_output=True,
        text=True,
    )
    ok = proc.returncode == 0
    if ok:
        got = np.loadtxt(out, dtype=np.float32, ndmin=2)
        exp = np.loadtxt(DATA / "example_expected.txt", dtype=np.float32, ndmin=2)
        ok = got.shape == exp.shape and np.array_equal(got, exp)
    print(f"{'OK  ' if ok else 'FAIL'} example      mode={mode:<12} exit={proc.returncode}")
    return ok


def run_empty(binary: Path, workdir: Path, mode: str) -> bool:
    """N=0: код возврата 0 и пустой выходной файл."""
    inp = workdir / "empty.txt"
    out = workdir / f"empty_{mode}.txt"
    inp.write_text("0 1.5\n")
    proc = subprocess.run(
        [str(binary), "--input", str(inp), "--output", str(out), "--mode", mode],
        capture_output=True,
        text=True,
    )
    ok = proc.returncode == 0 and out.exists() and out.stat().st_size == 0
    print(f"{'OK  ' if ok else 'FAIL'} n=0          mode={mode:<12} exit={proc.returncode}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/submission"))
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for mode in MODES:
            ok &= run_example(args.binary, workdir, mode)
            ok &= run_empty(args.binary, workdir, mode)
        for n in SIZES:
            for mode in MODES:
                ok &= run_case(args.binary, workdir, n, mode, rng)
    print("ALL OK" if ok else "SOME TESTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
