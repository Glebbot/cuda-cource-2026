"""Замеры времени: генерация входов, запуск submission в обоих режимах, таблицы для README.

Запуск из каталога лабораторной после сборки:
    python3 tests/bench.py [--binary build/submission] [--sizes 16 20 24]

Входы (N = 2^k пар) кешируются в build/bench/, сырой stdout сохраняется там же.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

SEED = 2026
ALPHA = 1.5
MODES = ["coalesced", "noncoalesced"]
# Число float-массивов, которые ядро читает и пишет: add и saxpy — 2 чтения + 1 запись, relu — 1 + 1.
ARRAYS_TOUCHED = {"add": 3, "saxpy": 3, "relu": 2}
LINE = re.compile(
    r"bench block=(\d+) metric=(\w+) median_ms=([\d.]+) min_ms=([\d.]+) max_ms=([\d.]+)"
)


def generate(path: Path, n: int) -> None:
    rng = np.random.default_rng(SEED)
    data = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    with path.open("w") as f:
        f.write(f"{n} {ALPHA}\n")
        np.savetxt(f, data, fmt="%.9g")


def run(binary: Path, inp: Path, out: Path, mode: str) -> str:
    proc = subprocess.run(
        [str(binary), "--input", str(inp), "--output", str(out), "--mode", mode],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"submission failed ({mode}, exit {proc.returncode}): {proc.stderr.strip()}")
    return proc.stdout


def parse(stdout: str) -> dict[int, dict[str, tuple[float, float, float]]]:
    result: dict[int, dict[str, tuple[float, float, float]]] = {}
    for m in LINE.finditer(stdout):
        block, metric = int(m.group(1)), m.group(2)
        result.setdefault(block, {})[metric] = (
            float(m.group(3)),
            float(m.group(4)),
            float(m.group(5)),
        )
    return result


def print_table(n: int, results: dict[str, dict[int, dict[str, tuple[float, float, float]]]]):
    print(f"\n### N = {n} (2^{n.bit_length() - 1})\n")
    print("Медиана, мс [мин–макс]; ГБ/с — эффективная пропускная способность ядра add.\n")
    header = ["Режим", "Блок", "H2D", "add", "saxpy", "relu", "D2H", "Полный цикл", "add, ГБ/с"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for mode in MODES:
        for block, metrics in sorted(results[mode].items()):
            cells = [mode, str(block)]
            for metric in ["h2d", "add", "saxpy", "relu", "d2h", "full"]:
                med, lo, hi = metrics[metric]
                cells.append(f"{med:.3f} [{lo:.3f}–{hi:.3f}]")
            add_ms = metrics["add"][0]
            gbps = ARRAYS_TOUCHED["add"] * n * 4 / (add_ms * 1e-3) / 1e9 if add_ms > 0 else 0.0
            cells.append(f"{gbps:.0f}")
            print("| " + " | ".join(cells) + " |")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/submission"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 20, 24], help="log2 N")
    parser.add_argument("--outdir", type=Path, default=Path("build/bench"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    for k in args.sizes:
        n = 1 << k
        inp = args.outdir / f"input_{k}.txt"
        if not inp.exists():
            print(f"generating {inp} ...", file=sys.stderr)
            generate(inp, n)
        results = {}
        for mode in MODES:
            print(f"running N=2^{k} mode={mode} ...", file=sys.stderr)
            stdout = run(args.binary, inp, args.outdir / f"output_{k}_{mode}.txt", mode)
            (args.outdir / f"stdout_{k}_{mode}.txt").write_text(stdout)
            results[mode] = parse(stdout)
            for line in stdout.splitlines():
                if line.startswith(("bench device=", "bench alloc_ms=")):
                    print(line)
        print_table(n, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
