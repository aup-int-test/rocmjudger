#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import glob
import time
import math
import signal
import pathlib
import subprocess
from collections import defaultdict

# ====== 可調參數 ======
USERS_FROM = 1          # 起始 user 編號（例如 1 -> user001）
USERS_TO = 25           # 結束 user 編號（例如 150 -> user150）
USER_FMT = "user{:03d}" # 帳號格式
BASE_HOME = "/home"
CONTEST_DIRNAME = "hip_programming_contest"
PROBLEMS = ["prefix_sum", "softmax", "apsp"]
TESTCASE_ROOT = "/home/amd/ccf2025_hidden/testcases"  # 你的測資根目錄（prefix_sum/softmax/apsp 各有一個資料夾）
TMP_OUTPUT_PRIMARY = "/dev/shm/tmp1"    # 優先寫到 /dev/shm
TMP_OUTPUT_FALLBACK = "/tmp/tmp_eval_out"
TIMEOUT_SEC = 360       # 每筆測資最長 6 分鐘
DETAIL_CSV = "1ccf2025_detailed_results.csv"
SUMMARY_CSV = "1ccf2025_summary_rankings.csv"
# softmax tolerance
SOFTMAX_ATOL = 1e-6
SOFTMAX_RTOL = 1e-5
SOFTMAX_MIN_DEN = 1e-12

# 白名單：不要跑測試的使用者（以整數學號表示，如 3 代表 user003）
SKIP_USER_IDS = {
    3, 17, 19, 21, 22, 24, 28, 29, 37, 38, 42, 45, 46, 47,
    55, 56, 73, 77, 96, 106, 108, 114, 115, 116, 117, 118,
    120, 121, 122, 125, 126, 133
}
# =====================

ANSI_RED = "\033[31m"
ANSI_BLUE = "\033[34m"
ANSI_YELLOW = "\033[33m"
ANSI_RESET = "\033[0m"

STATUS_AC = "AC"
STATUS_WA = "WA"
STATUS_TLE = "TLE"
STATUS_ERR = "ERROR"
STATUS_BUILD_FAIL = "BUILD_FAIL"
STATUS_SKIP = "SKIP"

def colorize(text, status):
    if status == STATUS_WA:
        return f"{ANSI_RED}{text}{ANSI_RESET}"
    if status == STATUS_TLE:
        return f"{ANSI_BLUE}{text}{ANSI_RESET}"
    if status in (STATUS_ERR, STATUS_BUILD_FAIL, STATUS_SKIP):
        return f"{ANSI_YELLOW}{text}{ANSI_RESET}"
    return text

def sorted_test_inputs(problem_dir):
    """回傳排序後的 .in 測資清單（依數字檔名排序，若非純數字則字典序）"""
    inputs = glob.glob(os.path.join(problem_dir, "*.in"))
    def keyfn(p):
        stem = pathlib.Path(p).stem
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem)
    return sorted(inputs, key=keyfn)

def kill_process_group(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass

def run_cmd(cmd, cwd=None, timeout=None, capture_output=False):
    """
    執行命令；為了 TLE 能完整殺 child tree，使用新 process group。
    回傳 (returncode, stdout_str, stderr_str, elapsed_sec, timed_out_bool, exception_or_None)
    """
    start = time.monotonic()
    timed_out = False
    exc = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            preexec_fn=os.setsid,  # 開新 pgid 方便整組殺
            text=True
        )
        try:
            outs, errs = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_process_group(proc)
            outs, errs = ("", "")
            rc = -9
        else:
            rc = proc.returncode
    except Exception as e:
        rc = -1
        outs, errs = ("", "")
        exc = e
    elapsed = time.monotonic() - start
    return rc, (outs or ""), (errs or ""), elapsed, timed_out, exc

def safe_make(problem_path):
    # make clean
    _rc, _o, _e, _t, _to, _ex = run_cmd(["make", "clean"], cwd=problem_path, capture_output=True)
    # make
    rc, o, e, t, to, ex = run_cmd(["make"], cwd=problem_path, capture_output=True)
    ok = (rc == 0 and not to and ex is None)
    return ok, rc, o, e

def compare_softmax_with_tolerance(expected_path, actual_path,
                                   atol=SOFTMAX_ATOL, rtol=SOFTMAX_RTOL, min_den=SOFTMAX_MIN_DEN):
    """
    以容忍度比較 softmax 輸出。返回 (ok, first_error_msg or None)
    規則:
      |got - exp| <= atol  或  |got - exp| / max(|exp|, min_den) <= rtol
    """
    try:
        with open(expected_path, "r") as fe, open(actual_path, "r") as fa:
            line_no = 0
            while True:
                el = fe.readline()
                al = fa.readline()
                if not el and not al:
                    break  # both EOF -> OK
                line_no += 1
                if not el or not al:
                    return False, f"Line {line_no}: line count mismatch"
                e_tokens = el.strip().split()
                a_tokens = al.strip().split()
                if len(e_tokens) != len(a_tokens):
                    return False, f"Line {line_no}: token count mismatch {len(e_tokens)} vs {len(a_tokens)}"
                for idx, (es, as_) in enumerate(zip(e_tokens, a_tokens)):
                    try:
                        ev = float(es)
                        av = float(as_)
                    except ValueError:
                        return False, f"Line {line_no}, col {idx+1}: non-numeric"
                    diff = abs(av - ev)
                    if diff <= atol:
                        continue
                    den = max(abs(ev), min_den)
                    rel = diff / den
                    if rel <= rtol:
                        continue
                    return False, f"Line {line_no}, col {idx+1}: diff={diff:.3e}, rel={rel:.3e} > rtol"
        return True, None
    except Exception as e:
        return False, f"exception: {e}"

def run_one_case(bin_path, in_path, tmp_out_path, timeout_sec, problem_name):
    """
    執行單筆測資：
    - stdout -> tmp_out_path
    - stderr -> DEVNULL（完全靜音，不噴終端）
    - 僅量測程式執行時間（不含比對時間）
    - 回傳 (status, run_time_sec, penalty_sec)
    """
    # 確保 tmp 檔路徑可用
    try:
        with open(tmp_out_path, "wb"):
            pass
    except Exception:
        tmp_out_path = TMP_OUTPUT_FALLBACK
        with open(tmp_out_path, "wb"):
            pass

    start = time.monotonic()
    timed_out = False
    rc = -1
    ex = None

    try:
        with open(tmp_out_path, "wb") as fout:
            proc = subprocess.Popen(
                [bin_path, in_path],
                cwd=os.path.dirname(bin_path),
                stdout=fout,                   # 關鍵：選手程式 stdout 寫到檔案，不上螢幕
                stderr=subprocess.DEVNULL,     # 關鍵：選手程式 stderr 丟棄
                preexec_fn=os.setsid,          # 方便 timeout 時整組殺掉
            )
            try:
                proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
                rc = -9
            else:
                rc = proc.returncode
    except Exception as e:
        ex = e

    elapsed = time.monotonic() - start

    if timed_out:
        return STATUS_TLE, elapsed, 60
    if ex is not None or rc != 0:
        return STATUS_ERR, elapsed, 60

    # 檢查對應的 .out
    test_dir = os.path.join(TESTCASE_ROOT, problem_name)
    stem = pathlib.Path(in_path).stem
    expected = os.path.join(test_dir, f"{stem}.out")
    if not os.path.exists(expected):
        # 缺 expected 視為錯（環境問題）
        return STATUS_ERR, elapsed, 60

    # softmax 用容忍；其他題用 diff（且把 diff 的輸出吞掉）
    if problem_name == "softmax":
        ok, _msg = compare_softmax_with_tolerance(expected, tmp_out_path)
        if ok:
            return STATUS_AC, elapsed, 0
        else:
            return STATUS_WA, elapsed, 60
    else:
        # 使用 diff 並忽略空白字元與換行差異
        diff_rc, _, _, _, _, _ = run_cmd(
            ["diff", "-qBw", expected, tmp_out_path],
            capture_output=True  # 攔截 diff 輸出，不印終端
        )
        if diff_rc == 0:
            return STATUS_AC, elapsed, 0
        elif diff_rc == 1:
            return STATUS_WA, elapsed, 60
        else:
            return STATUS_ERR, elapsed, 60

def rank_from_totals(user_totals):
    """
    user_totals: Dict[user] -> total_sec (float or None)
    回傳 Dict[user] -> rank（標準競賽排名法：同分同名次，之後名次跳過）
    """
    items = [(u, float('inf') if (v is None or math.isnan(v)) else v) for u, v in user_totals.items()]
    items.sort(key=lambda x: x[1])  # 依時間升冪
    ranks = {}
    last_time = None
    last_rank = 0
    count = 0
    for u, t in items:
        count += 1
        if last_time is None or t > last_time:
            last_rank = count
            last_time = t
        ranks[u] = last_rank
    return ranks

def main():
    # 準備輸出 CSV
    detail_rows = []
    per_user_per_problem_total = defaultdict(lambda: defaultdict(float))

    # 預先取得各題測資清單（確保每位 user 用同一批）
    problem_inputs = {}
    for prob in PROBLEMS:
        test_dir = os.path.join(TESTCASE_ROOT, prob)
        ins = sorted_test_inputs(test_dir)
        problem_inputs[prob] = ins
        print(f"[Info] Problem {prob}: found {len(ins)} testcases in {test_dir}")
    print()

    # 主迴圈：user → problem
    for uid in range(USERS_FROM, USERS_TO + 1):
        if uid in SKIP_USER_IDS:
            # 完全不評測，不寫入任何 CSV 紀錄，也不影響排名
            user_skipped = USER_FMT.format(uid)
            print(colorize(f"========== {user_skipped} ==========\n[Skip] in whitelist, skip evaluation.\n", STATUS_SKIP))
            continue

        user = USER_FMT.format(uid)
        user_dir = os.path.join(BASE_HOME, user, CONTEST_DIRNAME)
        if not os.path.isdir(user_dir):
            print(colorize(f"[Warn] {user}: directory not found: {user_dir}", STATUS_ERR))
            # 若整個使用者目錄不存在，三題都當作 ERROR（對應每筆測資 60s 罰）
            for prob in PROBLEMS:
                for in_path in problem_inputs[prob]:
                    test_id = pathlib.Path(in_path).stem
                    detail_rows.append({
                        "user": user,
                        "problem": prob,
                        "test_id": test_id,
                        "status": STATUS_ERR,
                        "time_sec": "",
                        "penalty_sec": 60,
                        "total_time_with_penalty_sec": 60
                    })
                    per_user_per_problem_total[user][prob] += 60
            continue

        print(f"========== {user} ==========")
        for prob in PROBLEMS:
            prob_path = os.path.join(user_dir, prob)
            bin_path = os.path.join(prob_path, prob)
            print(f"[Build] {user} / {prob} -> {prob_path}")

            if not os.path.isdir(prob_path):
                print(colorize(f"  [Error] Problem dir not found: {prob_path}", STATUS_ERR))
                # 當作整題不可測，對該題所有測資加 ERROR + 60
                for in_path in problem_inputs[prob]:
                    test_id = pathlib.Path(in_path).stem
                    detail_rows.append({
                        "user": user,
                        "problem": prob,
                        "test_id": test_id,
                        "status": STATUS_ERR,
                        "time_sec": "",
                        "penalty_sec": 60,
                        "total_time_with_penalty_sec": 60
                    })
                    per_user_per_problem_total[user][prob] += 60
                continue

            ok_build, rc_make, o_make, e_make = safe_make(prob_path)
            if not ok_build:
                print(colorize(f"  [Build Fail] rc={rc_make}", STATUS_BUILD_FAIL))
                # 整題 BUILD_FAIL：對所有測資記 60 罰
                for in_path in problem_inputs[prob]:
                    test_id = pathlib.Path(in_path).stem
                    detail_rows.append({
                        "user": user,
                        "problem": prob,
                        "test_id": test_id,
                        "status": STATUS_BUILD_FAIL,
                        "time_sec": "",
                        "penalty_sec": 60,
                        "total_time_with_penalty_sec": 60
                    })
                    per_user_per_problem_total[user][prob] += 60
                continue

            if not (os.path.isfile(bin_path) and os.access(bin_path, os.X_OK)):
                print(colorize(f"  [Error] Executable missing: {bin_path}", STATUS_ERR))
                for in_path in problem_inputs[prob]:
                    test_id = pathlib.Path(in_path).stem
                    detail_rows.append({
                        "user": user,
                        "problem": prob,
                        "test_id": test_id,
                        "status": STATUS_ERR,
                        "time_sec": "",
                        "penalty_sec": 60,
                        "total_time_with_penalty_sec": 60
                    })
                    per_user_per_problem_total[user][prob] += 60
                continue

            # 執行所有測資
            print(f"[Run] {user} / {prob}: {len(problem_inputs[prob])} cases")
            # 決定 tmp output 寫到 /dev/shm 或 /tmp
            tmp_out_path = TMP_OUTPUT_PRIMARY if os.path.isdir("/dev/shm") else TMP_OUTPUT_FALLBACK

            for in_path in problem_inputs[prob]:
                test_id = pathlib.Path(in_path).stem
                status, run_sec, penalty = run_one_case(bin_path, in_path, tmp_out_path, TIMEOUT_SEC, prob)
                # 終端列印（時間顏色）
                t_str = f"{run_sec:.3f}s" if isinstance(run_sec, (int, float)) and run_sec == run_sec else "-"
                print(f"  - case {test_id}: {colorize(status, status)}  time={colorize(t_str, status)}  +penalty={penalty}")

                total_with_penalty = (run_sec if isinstance(run_sec, (int, float)) else 0.0) + penalty
                detail_rows.append({
                    "user": user,
                    "problem": prob,
                    "test_id": test_id,
                    "status": status,
                    "time_sec": f"{run_sec:.6f}" if isinstance(run_sec, (int, float)) else "",
                    "penalty_sec": penalty,
                    "total_time_with_penalty_sec": f"{total_with_penalty:.6f}"
                })
                per_user_per_problem_total[user][prob] += total_with_penalty
            print()

    # --- 輸出明細 CSV ---
    detail_fieldnames = [
        "user", "problem", "test_id",
        "status", "time_sec", "penalty_sec", "total_time_with_penalty_sec"
    ]
    with open(DETAIL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fieldnames)
        writer.writeheader()
        # 排序：user, problem, test_id
        def sort_key(r):
            try:
                tid = int(r["test_id"])
            except:
                tid = r["test_id"]
            return (r["user"], r["problem"], tid)
        for row in sorted(detail_rows, key=sort_key):
            writer.writerow(row)
    print(f"[OK] Wrote detail results: {DETAIL_CSV}")

    # --- 計算每題總秒數與排名、overall ---
    users = sorted(per_user_per_problem_total.keys())
    per_problem_totals = {prob: {} for prob in PROBLEMS}
    overall_totals = {}

    for u in users:
        total_all = 0.0
        for prob in PROBLEMS:
            t = per_user_per_problem_total[u].get(prob, 0.0)
            per_problem_totals[prob][u] = t
            total_all += t
        overall_totals[u] = total_all

    # 各題排名
    per_problem_ranks = {prob: rank_from_totals(per_problem_totals[prob]) for prob in PROBLEMS}

    # --- 輸出彙總 CSV ---
    summary_fieldnames = [
        "user",
        "prefix_sum_total_sec", "prefix_sum_rank",
        "softmax_total_sec", "softmax_rank",
        "apsp_total_sec", "apsp_rank",
        "overall_total_sec"
    ]
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        for u in users:
            row = {
                "user": u,
                "prefix_sum_total_sec": f"{per_problem_totals['prefix_sum'][u]:.6f}",
                "prefix_sum_rank": per_problem_ranks['prefix_sum'][u],
                "softmax_total_sec": f"{per_problem_totals['softmax'][u]:.6f}",
                "softmax_rank": per_problem_ranks['softmax'][u],
                "apsp_total_sec": f"{per_problem_totals['apsp'][u]:.6f}",
                "apsp_rank": per_problem_ranks['apsp'][u],
                "overall_total_sec": f"{overall_totals[u]:.6f}",
            }
            writer.writerow(row)
    print(f"[OK] Wrote summary & rankings: {SUMMARY_CSV}")

    print("\nDone.")

if __name__ == "__main__":
    main()
