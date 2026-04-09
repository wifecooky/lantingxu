#!/usr/bin/env python3
"""兰亭序 · 全卷分段图 → 单字裁切 v2

改进：Otsu 自适应阈值 + 形态学清理 + 更可靠的列/字检测
"""

import sys
import os
import numpy as np
from PIL import Image, ImageFilter

EXPECTED_CHARS = 324


def stitch_segments(full_dir, prefix, nums):
    """水平拼接分段图"""
    segments = []
    for n in nums:
        path = os.path.join(full_dir, f"{prefix}{str(n).zfill(2)}.jpg")
        if os.path.exists(path):
            segments.append(Image.open(path))
    if not segments:
        raise FileNotFoundError(f"No segments found in {full_dir}")
    max_h = max(s.height for s in segments)
    total_w = sum(s.width for s in segments)
    stitched = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    x = 0
    for s in segments:
        y_offset = (max_h - s.height) // 2
        stitched.paste(s, (x, y_offset))
        x += s.width
    return stitched


def otsu_threshold(gray_arr):
    """Otsu 法自动选阈值"""
    hist, _ = np.histogram(gray_arr.ravel(), bins=256, range=(0, 256))
    total = gray_arr.size
    sum_all = np.sum(np.arange(256) * hist)
    sum_bg, w_bg, max_var, threshold = 0.0, 0, 0.0, 0
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_all - sum_bg) / w_fg
        var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var > max_var:
            max_var = var
            threshold = t
    return threshold


def smooth(arr, window=5):
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def find_regions(projection, min_gap, min_size, threshold_ratio=0.05):
    """从投影中找连续区域。阈值 = max * threshold_ratio"""
    th = projection.max() * threshold_ratio
    above = projection > th
    regions = []
    in_region = False
    start = 0
    for i in range(len(above)):
        if above[i] and not in_region:
            start = i
            in_region = True
        elif not above[i] and in_region:
            if i - start >= min_size:
                regions.append((start, i))
            in_region = False
    if in_region and len(above) - start >= min_size:
        regions.append((start, len(above)))
    # 合并间距太小的区域
    if not regions:
        return regions
    merged = [regions[0]]
    for r in regions[1:]:
        if r[0] - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    return merged


def crop_characters(stitched, output_dir, version_name, padding=6):
    gray = np.array(stitched.convert("L"))
    h, w = gray.shape

    # Otsu 阈值
    th = otsu_threshold(gray)
    print(f"  Otsu threshold: {th}")
    binary = (gray < th).astype(np.float32)

    # 1. 垂直投影找列（沿 y 轴求和 → 每个 x 位置的墨迹量）
    v_proj = smooth(binary.sum(axis=0), window=20)

    # 列的最小宽度约 = 图片高度 / 30（一列至少有一些字）
    col_min_size = max(15, h // 40)
    # 列间距至少几个像素
    col_min_gap = max(5, h // 200)

    columns = find_regions(v_proj, min_gap=col_min_gap, min_size=col_min_size, threshold_ratio=0.03)
    print(f"  Found {len(columns)} columns (expected ~28)")

    # 如果列太多（>50），说明噪声大，试着增大 min_gap
    if len(columns) > 50:
        col_min_gap = max(20, h // 50)
        columns = find_regions(v_proj, min_gap=col_min_gap, min_size=col_min_size, threshold_ratio=0.05)
        print(f"  Retried: found {len(columns)} columns")

    # 如果列太少（<10），列检测失败，用等分法
    if len(columns) < 10:
        print(f"  Column detection unreliable, falling back to grid-based approach")
        return crop_grid_based(stitched, gray, th, output_dir, version_name, padding)

    # 阅读顺序：右→左
    columns = list(reversed(columns))

    chars = []
    for ci, (cx0, cx1) in enumerate(columns):
        col_binary = binary[:, cx0:cx1]
        h_proj = smooth(col_binary.sum(axis=1), window=8)
        char_min_size = max(10, h // 50)
        char_min_gap = max(3, h // 200)
        char_bounds = find_regions(h_proj, min_gap=char_min_gap, min_size=char_min_size, threshold_ratio=0.05)

        for ry0, ry1 in char_bounds:
            x0 = max(0, cx0 - padding)
            x1 = min(w, cx1 + padding)
            y0 = max(0, ry0 - padding)
            y1 = min(h, ry1 + padding)
            chars.append((x0, y0, x1, y1))

    print(f"  Found {len(chars)} character regions")
    return save_chars(stitched, chars, output_dir)


def crop_grid_based(stitched, gray, threshold, output_dir, version_name, padding=6):
    """备用方案：基于已知的兰亭序布局做网格裁切

    兰亭序全文约 28 行，每行约 10-14 字。
    水平卷轴，文字从右到左排列。
    """
    h, w = gray.shape
    binary = (gray < threshold).astype(np.float32)

    # 找到有内容的区域（去掉左右空白边距）
    v_proj = smooth(binary.sum(axis=0), window=30)
    content_threshold = v_proj.max() * 0.02
    content_cols = np.where(v_proj > content_threshold)[0]
    if len(content_cols) == 0:
        print("  ERROR: No content found!")
        return 0
    content_left = content_cols[0]
    content_right = content_cols[-1]
    content_width = content_right - content_left

    # 找有内容的高度范围
    h_proj = smooth(binary.sum(axis=1), window=20)
    content_rows = np.where(h_proj > h_proj.max() * 0.02)[0]
    content_top = content_rows[0]
    content_bottom = content_rows[-1]
    content_height = content_bottom - content_top

    print(f"  Content area: x=[{content_left}:{content_right}] ({content_width}px), y=[{content_top}:{content_bottom}] ({content_height}px)")

    # 估算列数和每列字数
    # 兰亭序约 28 列（行），列宽 = content_width / 28
    n_cols = 28
    col_width = content_width / n_cols

    # 基于全文字数和列数，平均每列约 11-12 字
    # 但实际不均匀。我们用投影来微调每列的字数。
    chars = []

    for ci in range(n_cols):
        # 从右到左（阅读顺序）
        cx_right = content_right - ci * col_width
        cx_left = cx_right - col_width
        cx0 = int(max(0, cx_left))
        cx1 = int(min(w, cx_right))

        # 该列内的水平投影
        col_binary = binary[content_top:content_bottom, cx0:cx1]
        h_proj_col = smooth(col_binary.sum(axis=1), window=5)

        # 找字
        char_bounds = find_regions(h_proj_col, min_gap=3, min_size=max(8, content_height // 30), threshold_ratio=0.08)

        if len(char_bounds) == 0:
            # 如果找不到字，跳过这列（可能是空白/印章区域）
            continue

        for ry0, ry1 in char_bounds:
            abs_y0 = max(0, content_top + ry0 - padding)
            abs_y1 = min(h, content_top + ry1 + padding)
            x0 = max(0, cx0 - padding)
            x1 = min(w, cx1 + padding)
            chars.append((x0, abs_y0, x1, abs_y1))

    print(f"  Grid-based: found {len(chars)} character regions")
    return save_chars(stitched, chars, output_dir)


def make_square(img, size=200):
    w, h = img.size
    ratio = min(size / w, size / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    arr = np.array(img)
    edges = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    bg = tuple(edges.mean(axis=0).astype(int))
    square = Image.new("RGB", (size, size), bg)
    square.paste(img, ((size - new_w) // 2, (size - new_h) // 2))
    return square


def save_chars(stitched, chars, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # 清理旧的编号文件（保留 full/ 目录）
    for f in os.listdir(output_dir):
        if f.endswith(".jpg") and f[:3].isdigit():
            os.remove(os.path.join(output_dir, f))

    count = 0
    for i, (x0, y0, x1, y1) in enumerate(chars):
        crop = stitched.crop((x0, y0, x1, y1))
        crop = make_square(crop, 200)
        out_path = os.path.join(output_dir, f"{str(i + 1).zfill(3)}.jpg")
        crop.save(out_path, "JPEG", quality=92)
        count += 1
    return count


VERSIONS = {
    "chu": {"prefix": "csl", "nums": range(1, 20)},
    "yu": {"prefix": "ysn", "nums": range(1, 22)},
    "dingwu": {"prefix": "dw", "nums": list(range(1, 7)) + list(range(11, 18))},
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tools/crop_chars.py <version>")
        print("Versions: chu, yu, dingwu")
        sys.exit(1)

    version = sys.argv[1]
    if version not in VERSIONS:
        print(f"Unknown version: {version}")
        sys.exit(1)

    cfg = VERSIONS[version]
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_dir = os.path.join(base, "assets", version, "full")
    output_dir = os.path.join(base, "assets", version)

    print(f"Processing {version}...")
    stitched = stitch_segments(full_dir, cfg["prefix"], cfg["nums"])
    print(f"  Stitched: {stitched.size}")

    count = crop_characters(stitched, output_dir, version)
    print(f"  Done! {count} characters saved to {output_dir}/")


if __name__ == "__main__":
    main()
