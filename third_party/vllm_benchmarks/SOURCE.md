# Vendored vLLM benchmark scripts

| | |
|---|---|
| Source | https://github.com/vllm-project/vllm, `benchmarks/kernels/benchmark_device_communicators.py` and `benchmarks/kernels/benchmark_fused_collective.py` |
| Tag | `v0.30.0` |
| Commit | `ced6857afa0e` (`ced6857afa0ea7b2e3f0846a62e1394e90f15607`) |
| License | Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0). The SPDX headers of both files are kept. The rest of this repository is MIT. |

The scripts are not part of the vllm wheel, so they are copied here. `tpprof/vendored.py` loads them by file path
and runs them under `torchrun` (spec section 4.6, AM25, AM31):

- `benchmark_device_communicators.py` (M1, cross-check) is **unmodified**. The wrapper sets the module's
  `HIDDEN_SIZE` to 4096 and calls `vllm.distributed.parallel_state.init_distributed_environment(..., backend="gloo")`
  before `main()`. Without that call v0.30.0 drops FlashInfer from the results.
- `benchmark_fused_collective.py` (M2) is **modified** (AM31): `pandas` and the markdown report are removed, and
  `save_results_to_file` writes one JSON line per (num_tokens, op) to `--output-file`, with `"ms": null` for a
  failed op. The file is overwritten, not appended to, so a rerun never duplicates rows.

## SHA-256

- `49421414b5e9d51be04f95c4403b5c8ac8d014c0925d149af28babf6cafee6ca`  `upstream/benchmark_device_communicators.py`
- `49421414b5e9d51be04f95c4403b5c8ac8d014c0925d149af28babf6cafee6ca`  `vendored/benchmark_device_communicators.py`
- `8edf7657c04ad6cb32f8f3cf45888c0d8d582f8866066b74cf59b74a79444c78`  `upstream/benchmark_fused_collective.py`
- `12be53ee72e5e29c0a7ace33600024b82cb8269a8c290cafe77f75499935c108`  `vendored/benchmark_fused_collective.py`

## Diff of `benchmark_fused_collective.py`

```diff
--- a/benchmarks/kernels/benchmark_fused_collective.py
+++ b/third_party/vllm_benchmarks/benchmark_fused_collective.py
@@ -1,5 +1,6 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Modified for vllm-tp2-profiling: see third_party/vllm_benchmarks/SOURCE.md.
 
 """
 Benchmark for FlashInfer fused collective operations vs standard operations.
@@ -18,10 +19,10 @@
 
 import argparse
 import itertools
+import json
 import os
 import time
 
-import pandas as pd
 import torch  # type: ignore
 import torch.distributed as dist  # type: ignore
 
@@ -852,82 +853,32 @@
         print(
             f"{result['operation']:<50} {time_display:<12} {result['speedup_str']:<10}"
         )
-
-
-def format_results_markdown(
-    all_results: list[dict], world_size: int, args: argparse.Namespace
-) -> str:
-    """Format all benchmark results as markdown."""
-    lines: list[str] = []
-    lines.append("# FlashInfer Fused Collective Operations Benchmark Results")
-    lines.append("")
-    lines.append(f"**World Size:** {world_size}  ")
-    lines.append(f"**Hidden Dimension:** {args.hidden_dim}  ")
-    lines.append(f"**Warmup Iterations:** {args.warmup}  ")
-    lines.append(f"**Benchmark Trials:** {args.trials}  ")
-    modes = ",".join(all_results[0]["quant_modes"]) if all_results else "N/A"
-    lines.append(f"**Quantization Modes:** {modes}  ")
-    lines.append("")
-    lines.append("---")
-    lines.append("")
-
-    for entry in all_results:
-        num_tokens = entry["num_tokens"]
-        dtype = entry["dtype"]
-        use_residual = entry["use_residual"]
-        results_dict = entry["results"]
-        input_size_mb = entry["input_size_mb"]
-        residual_str = "with residual" if use_residual else "no residual"
 
-        lines.append(
-            f"## Configuration: num_tokens={num_tokens}, dtype={dtype}, {residual_str}"
-        )
-        lines.append(f"**Input Size:** {input_size_mb:.2f} MB")
-        lines.append("")
 
-        prepared = prepare_results_with_speedups(results_dict)
-        # Build DataFrame for markdown export
-        rows = [
-            {
-                "Operation": r["operation"].replace("_", " ").title(),
-                "Time (ms)": r["time_str"],
-                "Speedup": r["speedup_str"],
-            }
-            for r in prepared
-        ]
-        df = pd.DataFrame(rows)
-        if df.empty:
-            lines.append("No results.")
-        else:
-            lines.append(df.to_markdown(index=False))
-        lines.append("")
-
-    return "\n".join(lines)
-
-
 def save_results_to_file(
     all_results: list[dict], world_size: int, args: argparse.Namespace, rank: int
 ):
-    """Save benchmark results to markdown file (only on rank 0)."""
+    """Save benchmark results as JSON lines, one per (num_tokens, op) (only on rank 0).
+
+    A failed op (time ``inf``) is written with ``"ms": null``.
+    """
     if rank != 0:
         return
 
-    if not all_results:
-        logger.warning("No results to save")
-        return
+    with open(args.output_file, "w") as f:
+        for entry in all_results:
+            for op_name, time_ms in entry["results"].items():
+                line = {
+                    "num_tokens": entry["num_tokens"],
+                    "hidden_dim": entry["hidden_dim"],
+                    "dtype": entry["dtype"],
+                    "use_residual": entry["use_residual"],
+                    "op": op_name,
+                    "ms": None if time_ms == float("inf") else time_ms,
+                }
+                f.write(json.dumps(line) + "\n")
 
-    output_path = args.output_file
 
-    try:
-        markdown_content = format_results_markdown(all_results, world_size, args)
-
-        with open(output_path, "a") as f:
-            f.write(markdown_content)
-
-    except Exception as e:
-        logger.error("Failed to save results to file: %s", e)
-
-
 def main():
     parser = argparse.ArgumentParser(
         description="Benchmark fused collective operations"
@@ -975,9 +926,7 @@
     parser.add_argument(
         "--output-file",
         type=str,
-        help="""Output file path for markdown results 
-                (default: benchmark_results_<timestamp>.md)
-        """,
+        help="Output file path for JSON-lines results (one line per num_tokens and op)",
     )
 
     parser.add_argument(
```
