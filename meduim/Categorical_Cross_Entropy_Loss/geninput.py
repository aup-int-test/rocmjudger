#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import random

# ====== 可調參數（可依需求微調） ======
SEED = 77777
OUT_DIR = "testcases"
EPS = 1e-12  # 極小值，避免出現 0 機率
FMT = "{:.8f}"  # 機率輸出格式

def create_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def softmax(logits):
    """Numerically stable softmax."""
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]

def clamp_probs(ps, eps=EPS):
    """把機率下限夾到 eps，並重新正規化為 1。"""
    ps = [max(p, eps) for p in ps]
    s = sum(ps)
    return [p / s for p in ps]

def write_case(filename, batch_size, num_classes, preds, labels):
    with open(filename, "w") as f:
        f.write(f"{batch_size} {num_classes}\n")
        # predictions
        for i in range(batch_size):
            f.write(" ".join(FMT.format(p) for p in preds[i]) + "\n")
        # one-hot labels
        for i in range(batch_size):
            f.write(" ".join(str(int(x)) for x in labels[i]) + "\n")

def gen_random_labels(batch_size, num_classes, skew=None):
    """
    產生 one-hot 標籤。
    - skew=None: 各類別均勻機率。
    - skew=(p0, ..., pC-1): 自訂類別分佈（會正規化）。
    """
    if skew is None:
        dist = [1.0 / num_classes] * num_classes
    else:
        s = sum(skew)
        dist = [max(x, 0.0) / s for x in skew]

    labels = []
    for _ in range(batch_size):
        r = random.random()
        acc = 0.0
        cls = 0
        for j, p in enumerate(dist):
            acc += p
            if r <= acc:
                cls = j
                break
        onehot = [0] * num_classes
        onehot[cls] = 1
        labels.append(onehot)
    return labels

def gen_preds_from_logits(batch_size, num_classes, logit_gen_fn):
    preds = []
    for _ in range(batch_size):
        logits = logit_gen_fn(num_classes)
        ps = softmax(logits)
        ps = clamp_probs(ps, EPS)
        preds.append(ps)
    return preds

def logit_random_gaussian(num_classes, mu=0.0, sigma=1.0):
    return [random.gauss(mu, sigma) for _ in range(num_classes)]

def logit_almost_correct(num_classes, true_cls, high=6.0, low=-6.0):
    logits = [low] * num_classes
    logits[true_cls] = high
    return logits

def logit_almost_wrong(num_classes, wrong_cls, high=6.0, low=-6.0):
    logits = [low] * num_classes
    logits[wrong_cls] = high
    return logits

def gen_uniform_preds(batch_size, num_classes):
    row = [1.0 / num_classes] * num_classes
    return [row[:] for _ in range(batch_size)]

def gen_near_zero_edge_preds(batch_size, num_classes, min_p=EPS, max_p=1e-6):
    """
    產生包含極小機率的分佈，再正規化，避免完全 0。
    """
    preds = []
    for _ in range(batch_size):
        xs = [random.uniform(min_p, max_p) for _ in range(num_classes)]
        s = sum(xs)
        ps = [x / s for x in xs]
        ps = clamp_probs(ps, EPS)
        preds.append(ps)
    return preds

def size_report(path):
    if os.path.exists(path):
        sz = os.path.getsize(path)
        print(f"  {path}: {sz:,} bytes")

def main():
    random.seed(SEED)
    create_dir(OUT_DIR)

    # ========== 1. 小型隨機 ==========
    # B=4, C=3：快速手算/驗證
    B, C = 4, 3
    labels = gen_random_labels(B, C)
    preds = gen_preds_from_logits(B, C, lambda C: logit_random_gaussian(C, 0.0, 1.0))
    fn = os.path.join(OUT_DIR, "1.in")
    write_case(fn, B, C, preds, labels)
    size_report(fn)

    # ========== 2. 一般隨機 ==========
    # B=32, C=10：常見分類設定
    B, C = 32, 10
    labels = gen_random_labels(B, C)
    preds = gen_preds_from_logits(B, C, lambda C: logit_random_gaussian(C, 0.0, 1.5))
    fn = os.path.join(OUT_DIR, "2.in")
    write_case(fn, B, C, preds, labels)
    size_report(fn)

    # ========== 3. 幾乎全對（低 loss） ==========
    # 高信心預測在正確類別
    B, C = 64, 7
    labels = gen_random_labels(B, C)
    preds = []
    for i in range(B):
        true_cls = labels[i].index(1)
        ps = softmax(logit_almost_correct(C, true_cls, high=8.0, low=-8.0))
        preds.append(clamp_probs(ps, EPS))
    fn = os.path.join(OUT_DIR, "3.in")
    write_case(fn, B, C, preds, labels)
    size_report(fn)

    # ========== 4. 幾乎全錯（高 loss） ==========
    # 將最高機率硬壓在錯誤類別
    B, C = 64, 7
    labels = gen_random_labels(B, C)
    preds = []
    for i in range(B):
        true_cls = labels[i].index(1)
        wrong_cls = (true_cls + 1) % C
        ps = softmax(logit_almost_wrong(C, wrong_cls, high=8.0, low=-8.0))
        preds.append(clamp_probs(ps, EPS))
    fn = os.path.join(OUT_DIR, "4.in")
    write_case(fn, B, C, preds, labels)
    size_report(fn)

    # ========== 5. 高熵（均勻） ==========
    # 每筆預測都是完全均勻分佈
    B, C = 16, 5
    labels = gen_random_labels(B, C)
    preds = gen_uniform_preds(B, C)
    fn = os.path.join(OUT_DIR, "5.in")
    write_case(fn, B, C, preds, labels)
    size_report(fn)

    # ========== 6. 極小機率邊界測試 ==========
    # 機率非常接近 0，測試 log 與 clamp 的穩定性
    B, C = 16, 9
    labels = gen_random_labels(B, C)
    preds = gen_near_zero_edge_preds(B, C, min_p=EPS, max_p=1e-6)
    fn = os.path.join(OUT_DIR, "6.in")
    write_case(fn, B, C, preds, labels)
    size_report(fn)

    # ========== 7. 類別不均（標籤偏斜） ==========
    # 例如 70% 都是類別 0，其餘平均
    B, C = 128, 8
    skew = [0.7] + [0.3 / (C - 1)] * (C - 1)
    labels = gen_random_labels(B, C, skew=skew)
    # 模型預測也稍微偏斜，但不是極端
    preds = gen_preds_from_logits(B, C, lambda C: [random.gauss(0.2 if j == 0 else 0.0, 1.0) for j in range(C)])
    # clamp 一次
    preds = [clamp_probs(p, EPS) for p in preds]
    fn = os.path.join(OUT_DIR, "7.in")
    write_case(fn, B, C, preds, labels)
    size_report(fn)

    # ========== 8. 較大批次 ==========
    # 控制在合理檔案大小；可依硬體再放大
    B, C = 16000, 4000
    labels = gen_random_labels(B, C)
    preds = gen_preds_from_logits(B, C, lambda C: logit_random_gaussian(C, 0.0, 1.2))
    fn = os.path.join(OUT_DIR, "8.in")
    write_case(fn, B, C, preds, labels)
    size_report(fn)

if __name__ == "__main__":
    main()
