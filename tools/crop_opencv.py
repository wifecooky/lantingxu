#!/usr/bin/env python3
"""兰亭序 · OpenCV 单字裁切

利用 HSV 色彩过滤去除红色印章，自适应阈值提取墨迹，
垂直投影检测列，水平投影检测每列中的字。

用法: python3 tools/crop_opencv.py <version> [--preview] [--debug]
"""

import sys
import os
import cv2
import numpy as np
from PIL import Image

EXPECTED_CHARS = 324

VERSIONS = {
    "shenlong": {
        "source": "/tmp/shenlong_hires.jpg",
        "text_x": (0.70, 0.835),
        "text_y": (0.04, 0.96),
    },
    "chu": {
        "prefix": "csl", "nums": range(1, 20),
        "text_x": (0.100, 0.314),
        "text_y": (0.04, 0.93),
    },
    "yu": {
        "prefix": "ysn", "nums": range(1, 22),
        "text_x": (0.62, 0.925),
        "text_y": (0.04, 0.93),
    },
    "dingwu": {
        "prefix": "dw", "nums": list(range(1, 7)) + list(range(11, 18)),
        "text_x": (0.04, 0.96),
        "text_y": (0.04, 0.96),
    },
}


def stitch_segments(full_dir, prefix, nums):
    segments = []
    for n in nums:
        path = os.path.join(full_dir, "%s%02d.jpg" % (prefix, n))
        if os.path.exists(path):
            segments.append(cv2.imread(path))
    if not segments:
        raise FileNotFoundError("No segments in %s" % full_dir)
    max_h = max(s.shape[0] for s in segments)
    result = []
    for s in segments:
        if s.shape[0] < max_h:
            pad = np.full((max_h - s.shape[0], s.shape[1], 3), 255, dtype=np.uint8)
            s = np.vstack([s, pad])
        result.append(s)
    return np.hstack(result)


def remove_red_seals(img):
    """用 HSV 过滤去除红色印章"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 红色在 HSV 中分两段
    mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (160, 50, 50), (180, 255, 255))
    red_mask = mask1 | mask2
    # 膨胀红色区域，确保印章完全覆盖
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.dilate(red_mask, kernel, iterations=2)
    # 用背景色填充红色区域
    result = img.copy()
    result[red_mask > 0] = [230, 220, 200]  # 近似纸张背景色
    return result


def extract_ink(img):
    """提取黑色墨迹的二值图"""
    # 先去红色印章
    clean = remove_red_seals(img)
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    # 自适应阈值
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=31, C=15
    )
    # 去噪：开运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


def find_columns_cv(binary, min_col_width=20):
    """用垂直投影找列"""
    h, w = binary.shape
    v_proj = binary.sum(axis=0).astype(float)
    # 平滑
    kernel_size = max(5, w // 200)
    v_proj = np.convolve(v_proj, np.ones(kernel_size) / kernel_size, mode='same')

    # 找连续墨迹区域
    threshold = v_proj.max() * 0.03
    above = v_proj > threshold

    regions = []
    in_r = False
    start = 0
    for i in range(len(above)):
        if above[i] and not in_r:
            start = i
            in_r = True
        elif not above[i] and in_r:
            if i - start >= min_col_width:
                regions.append((start, i))
            in_r = False
    if in_r and len(above) - start >= min_col_width:
        regions.append((start, len(above)))

    # 合并太近的区域
    if regions:
        merged = [regions[0]]
        min_gap = max(5, min_col_width // 3)
        for r in regions[1:]:
            if r[0] - merged[-1][1] < min_gap:
                merged[-1] = (merged[-1][0], r[1])
            else:
                merged.append(r)
        regions = merged

    return regions


def find_chars_in_column(binary, x0, x1, min_char_height=15):
    """在一列中用水平投影找字"""
    col = binary[:, x0:x1]
    h_proj = col.sum(axis=1).astype(float)
    kernel_size = max(3, col.shape[0] // 100)
    h_proj = np.convolve(h_proj, np.ones(kernel_size) / kernel_size, mode='same')

    threshold = h_proj.max() * 0.05
    above = h_proj > threshold

    regions = []
    in_r = False
    start = 0
    for i in range(len(above)):
        if above[i] and not in_r:
            start = i
            in_r = True
        elif not above[i] and in_r:
            if i - start >= min_char_height:
                regions.append((start, i))
            in_r = False
    if in_r and len(above) - start >= min_char_height:
        regions.append((start, len(above)))

    # 合并太近的区域
    if regions:
        merged = [regions[0]]
        min_gap = max(3, min_char_height // 3)
        for r in regions[1:]:
            if r[0] - merged[-1][1] < min_gap:
                merged[-1] = (merged[-1][0], r[1])
            else:
                merged.append(r)
        regions = merged

    return regions


def make_square(img, size=300):
    h, w = img.shape[:2]
    ratio = min(size / w, size / h) * 0.85  # 留点边距
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    # 背景色取边缘像素均值
    edges = np.concatenate([resized[0], resized[-1], resized[:, 0], resized[:, -1]])
    bg = edges.mean(axis=0).astype(np.uint8)
    square = np.full((size, size, 3), bg, dtype=np.uint8)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    square[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return square


def crop_version(img, output_dir, version_name, cfg, preview=False, debug=False):
    h, w = img.shape[:2]
    print("  图片: %dx%d" % (w, h))

    # 裁出正文区域
    tx = cfg.get("text_x", (0, 1))
    ty = cfg.get("text_y", (0, 1))
    x0, x1 = int(w * tx[0]), int(w * tx[1])
    y0, y1 = int(h * ty[0]), int(h * ty[1])
    text_img = img[y0:y1, x0:x1]
    th, tw = text_img.shape[:2]
    print("  正文区: %dx%d" % (tw, th))

    # 提取墨迹
    binary = extract_ink(text_img)

    if debug:
        cv2.imwrite("/tmp/%s_binary.jpg" % version_name, binary)
        print("  二值图: /tmp/%s_binary.jpg" % version_name)

    # 找列
    columns = find_columns_cv(binary, min_col_width=max(10, tw // 80))
    # 从右到左排（阅读序）
    columns.reverse()
    print("  检测到 %d 列" % len(columns))

    # 找每列中的字
    all_chars = []  # (x0_abs, y0_abs, x1_abs, y1_abs)
    for ci, (cx0, cx1) in enumerate(columns):
        chars = find_chars_in_column(binary, cx0, cx1, min_char_height=max(10, th // 30))
        for (cy0, cy1) in chars:
            # 转为原图绝对坐标
            all_chars.append((x0 + cx0, y0 + cy0, x0 + cx1, y0 + cy1))
        if debug:
            print("  列%d: x=[%d:%d] %d字" % (ci + 1, cx0, cx1, len(chars)))

    print("  检测到 %d 个字（期望 %d）" % (len(all_chars), EXPECTED_CHARS))

    # 预览
    if preview:
        preview_img = img.copy()
        for i, (ax0, ay0, ax1, ay1) in enumerate(all_chars):
            color = (0, 255, 0) if i < EXPECTED_CHARS else (0, 0, 255)
            cv2.rectangle(preview_img, (ax0, ay0), (ax1, ay1), color, 2)
            # 编号
            cv2.putText(preview_img, str(i + 1), (ax0, ay0 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        out = "/tmp/%s_preview.jpg" % version_name
        cv2.imwrite(out, preview_img)
        print("  预览: %s" % out)
        return len(all_chars)

    # 裁切
    os.makedirs(output_dir, exist_ok=True)
    for f in os.listdir(output_dir):
        if f.endswith(".jpg") and f[:3].isdigit():
            os.remove(os.path.join(output_dir, f))

    count = min(len(all_chars), EXPECTED_CHARS)
    for i in range(count):
        ax0, ay0, ax1, ay1 = all_chars[i]
        # 加点 padding
        pad = 4
        crop = img[max(0, ay0 - pad):min(h, ay1 + pad),
                    max(0, ax0 - pad):min(w, ax1 + pad)]
        crop = make_square(crop, 300)
        out_path = os.path.join(output_dir, "%03d.jpg" % (i + 1))
        cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print("  裁切完成: %d 个字" % count)
    return count


def main():
    if len(sys.argv) < 2:
        print("用法: python3 tools/crop_opencv.py <version> [--preview] [--debug]")
        print("版本: shenlong, chu, yu, dingwu")
        sys.exit(1)

    version = sys.argv[1]
    preview = "--preview" in sys.argv
    debug = "--debug" in sys.argv

    if version not in VERSIONS:
        print("未知版本: %s" % version)
        sys.exit(1)

    cfg = VERSIONS[version]
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base, "assets", version)

    # 加载图片
    if "source" in cfg:
        print("加载: %s" % cfg["source"])
        img = cv2.imread(cfg["source"])
    else:
        full_dir = os.path.join(base, "assets", version, "full")
        stitched_path = os.path.join(full_dir, "_stitched.jpg")
        if os.path.exists(stitched_path):
            print("加载: %s" % stitched_path)
            img = cv2.imread(stitched_path)
        else:
            print("拼接分段图...")
            img = stitch_segments(full_dir, cfg["prefix"], cfg["nums"])
            cv2.imwrite(stitched_path, img)

    print("处理 %s..." % version)
    count = crop_version(img, output_dir, version, cfg, preview, debug)
    if not preview:
        print("完成! %d 个字" % count)


if __name__ == "__main__":
    main()
