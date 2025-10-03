# CCC/CCF2025 Evaluation Runner

This repository provides the evaluation script `run.py` for the **CCF2025 HIP Programming Contest**.
The script will iterate through each user’s contest directory and execute the evaluation process for all problems.

In our competition setup, we copy this script into **six separate instances** and launch them simultaneously on different GPUs using `HIP_VISIBLE_DEVICES`.
This way, 150 users are split into six batches and evaluated in parallel across six GPUs.

---

## Usage

```bash
python3 run.py
```

---

## Adjustable Parameters

The following parameters can be configured directly inside `run.py`:

* **`USERS_FROM`**
  Starting user index (e.g., `1` → `user001`).

* **`USERS_TO`**
  Ending user index (e.g., `150` → `user150`).

* **`USER_FMT`**
  Username formatting string.
  Default: `"user{:03d}"` → produces `user001`, `user002`, ...

* **`BASE_HOME`**
  Base directory for user home paths.
  Example: `/home`

* **`CONTEST_DIRNAME`**
  Contest working directory name inside each user’s home.
  Example: `hip_programming_contest`

* **`PROBLEMS`**
  List of problem names to evaluate.
  Example: `["prefix_sum", "softmax", "apsp"]`

* **`TESTCASE_ROOT`**
  Root directory for hidden testcases.
  Each problem has its own subfolder:

  ```
  /home/amd/ccf2025_hidden/testcases/prefix_sum
  /home/amd/ccf2025_hidden/testcases/softmax
  /home/amd/ccf2025_hidden/testcases/apsp
  ```

* **`TMP_OUTPUT_PRIMARY`**
  Primary temporary output directory (preferably in memory, e.g., `/dev/shm/tmp1`).

* **`TMP_OUTPUT_FALLBACK`**
  Fallback temporary output directory (e.g., `/tmp/tmp_eval_out`).

* **`TIMEOUT_SEC`**
  Maximum time allowed per testcase (default: 360 seconds = 6 minutes).
  Testcases exceeding this limit are marked as **TLE**.

* **`DETAIL_CSV`**
  Path of the detailed results CSV file.
  Example: `1ccf2025_detailed_results.csv`

* **`SUMMARY_CSV`**
  Path of the summary rankings CSV file.
  Example: `1ccf2025_summary_rankings.csv`

---

## Notes

* Temporary output is written to `/dev/shm` whenever possible to reduce I/O overhead.
