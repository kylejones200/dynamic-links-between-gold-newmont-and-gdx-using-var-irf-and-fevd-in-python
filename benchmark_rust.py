#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import var1_irf  # noqa: E402

def main() -> None:
    phi, shock, steps = 0.85, 1.0, 24
    t0 = time.perf_counter()
    for _ in range(200):
        var1_irf(phi, shock, steps)
    py_s = time.perf_counter() - t0
    try:
        import dynamic_links_between_gold_newmont_and_gdx_using_var_irf_and_fevd_in_python_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(phi, shock, steps, 5000)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    np.testing.assert_allclose(
        var1_irf(phi, shock, steps), np.asarray(rs.var1_irf_py(phi, shock, steps)), rtol=1e-12
    )
    print("Correctness: OK")

if __name__ == "__main__":
    main()
