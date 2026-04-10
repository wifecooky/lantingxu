#!/usr/bin/env python3
"""兰亭序 · 引导式单字裁切 v3

利用已知的全文布局（每列字数）引导裁切。
步骤：
  1. 加载拼接图，转灰度+二值化
  2. 用垂直投影找列边界（右→左阅读序）
  3. 已知每列字数，在每列内按字数等分
  4. 裁出每个字，居中放入正方形，输出

用法: python3 tools/crop_guided.py <version> [--preview]
"""

import sys
import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 兰亭序全文 324 字（不含标点）
EXPECTED_CHARS = 324
# 标准列数约 28（不同版本略有差异，但总字数不变）
DEFAULT_N_COLS = 28

# 每个版本的配置：分段图前缀/编号 + 正文区域（相对比例）
# shenlong 使用单独的高清全卷图
VERSIONS = {
    "shenlong": {
        "source": "/tmp/shenlong_hires.jpg",  # Wikimedia 高清图
        "text_x": (0.70, 0.835),
        "text_y": (0.04, 0.96),
        "n_cols": 28,
    },
    "chu": {
        "prefix": "csl", "nums": range(1, 20),
        # 褚摹本正文集中在 csl03-06 (x=10%-31%)
        "text_x": (0.100, 0.314),
        "text_y": (0.04, 0.93),
        "n_cols": 28,
        "ltr": True,
    },
    "yu": {
        "prefix": "ysn", "nums": range(1, 22),
        "text_x": (0.62, 0.925),
        "text_y": (0.04, 0.93),
        "n_cols": 28,
        "ltr": True,
    },
    "dingwu": {
        "prefix": "dw", "nums": list(range(1, 7)) + list(range(11, 18)),
        "text_x": (0.04, 0.96),
        "text_y": (0.04, 0.96),
        "n_cols": 28,
        "ltr": True,
    },
}


def stitch_segments(full_dir, prefix, nums):
    """水平拼接分段图"""
    segments = []
    for n in nums:
        path = os.path.join(full_dir, "%s%02d.jpg" % (prefix, n))
        if os.path.exists(path):
            segments.append(Image.open(path))
    if not segments:
        raise FileNotFoundError("No segments in %s" % full_dir)
    max_h = max(s.height for s in segments)
    total_w = sum(s.width for s in segments)
    stitched = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    x = 0
    for s in segments:
        stitched.paste(s, (x, (max_h - s.height) // 2))
        x += s.width
    return stitched


def otsu_threshold(gray_arr):
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


def find_text_region(gray, threshold):
    """找到正文区域（去掉印章、题跋等边缘区域）"""
    h, w = gray.shape
    binary = (gray < threshold).astype(np.float32)

    # 红色印章在灰度图中偏亮，正文墨迹偏暗
    # 用更严格的阈值（只保留深色墨迹）来找正文区域
    dark_th = max(80, threshold - 40)
    dark_binary = (gray < dark_th).astype(np.float32)

    # 垂直投影找正文左右边界
    v_proj = smooth(dark_binary.sum(axis=0), window=50)
    content_th = v_proj.max() * 0.08
    cols = np.where(v_proj > content_th)[0]
    if len(cols) < 10:
        return 0, w, 0, h

    # 水平投影找正文上下边界
    h_proj = smooth(dark_binary.sum(axis=1), window=30)
    rows = np.where(h_proj > h_proj.max() * 0.08)[0]

    return cols[0], cols[-1], rows[0], rows[-1]


def find_columns(binary, x0, x1, y0, y1, n_cols, ltr=False):
    """在正文区域中找到 n_cols 个列的边界。ltr=True 时从左到右为阅读序"""
    region = binary[y0:y1, x0:x1]
    h, w = region.shape

    # 垂直投影
    v_proj = smooth(region.sum(axis=0), window=15)

    # 找波谷（列间空隙）
    # 先尝试自动检测列边界
    threshold = v_proj.max() * 0.03
    above = v_proj > threshold

    # 找连续的"有墨迹"区域
    regions = []
    in_region = False
    start = 0
    for i in range(len(above)):
        if above[i] and not in_region:
            start = i
            in_region = True
        elif not above[i] and in_region:
            if i - start > w // (n_cols * 3):  # 最小列宽
                regions.append((start, i))
            in_region = False
    if in_region and len(above) - start > w // (n_cols * 3):
        regions.append((start, len(above)))

    # 合并间距太小的区域
    if regions:
        merged = [regions[0]]
        min_gap = w // (n_cols * 2)
        for r in regions[1:]:
            if r[0] - merged[-1][1] < min_gap:
                merged[-1] = (merged[-1][0], r[1])
            else:
                merged.append(r)
        regions = merged

    print("  自动检测到 %d 个列区域（期望 %d）" % (len(regions), n_cols))

    # 如果检测到的列数和期望接近，就用检测结果
    if abs(len(regions) - n_cols) <= 3:
        # 微调：如果多了就合并最窄的，少了就拆分最宽的
        while len(regions) > n_cols and len(regions) > 1:
            # 合并最窄的相邻对
            widths = [r[1] - r[0] for r in regions]
            min_idx = widths.index(min(widths))
            if min_idx > 0:
                regions[min_idx - 1] = (regions[min_idx - 1][0], regions[min_idx][1])
                regions.pop(min_idx)
            else:
                regions[0] = (regions[0][0], regions[1][1])
                regions.pop(1)

        while len(regions) < n_cols:
            # 拆分最宽的列
            widths = [r[1] - r[0] for r in regions]
            max_idx = widths.index(max(widths))
            r = regions[max_idx]
            mid = (r[0] + r[1]) // 2
            regions[max_idx] = (r[0], mid)
            regions.insert(max_idx + 1, (mid, r[1]))

        # 转换为绝对坐标
        cols = [(x0 + r[0], x0 + r[1]) for r in regions]
        if not ltr:
            cols.reverse()  # 右→左排序
        return cols

    # 检测完全不可靠，用等分法
    print("  列检测不可靠，使用等分法")
    col_width = w / n_cols
    cols = []
    if ltr:
        # 拼接图从左到右 = 阅读序
        for i in range(n_cols):
            cx0 = x0 + int(i * col_width)
            cx1 = x0 + int((i + 1) * col_width)
            cols.append((cx0, cx1))
    else:
        # 拼接图从右到左 = 阅读序
        for i in range(n_cols):
            cx0 = x0 + int((n_cols - 1 - i) * col_width)
            cx1 = x0 + int((n_cols - i) * col_width)
            cols.append((cx0, cx1))
    return cols


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


def crop_version(stitched, output_dir, version_name, cfg, preview=False):
    gray = np.array(stitched.convert("L"))
    h, w = gray.shape
    th = otsu_threshold(gray)
    binary = (gray < th).astype(np.float32)
    print("  图片尺寸: %dx%d, Otsu 阈值: %d" % (w, h, th))

    # 构建分段列表：每段有自己的区域和列数
    segments = cfg.get("segments", None)
    if segments:
        seg_list = []
        for seg in segments:
            tx0 = int(w * seg["x"][0])
            tx1 = int(w * seg["x"][1])
            ty0 = int(h * seg["y"][0])
            ty1 = int(h * seg["y"][1])
            seg_list.append((tx0, tx1, ty0, ty1, seg["cols"]))
    else:
        tx_pct = cfg.get("text_x", (0.0, 1.0))
        ty_pct = cfg.get("text_y", (0.0, 1.0))
        tx0 = int(w * tx_pct[0])
        tx1 = int(w * tx_pct[1])
        ty0 = int(h * ty_pct[0])
        ty1 = int(h * ty_pct[1])
        seg_list = [(tx0, tx1, ty0, ty1, cfg.get("n_cols", DEFAULT_N_COLS))]

    ltr = cfg.get("ltr", False)
    total_cols = sum(s[4] for s in seg_list)
    # 每列平均字数
    avg = EXPECTED_CHARS // total_cols
    remainder = EXPECTED_CHARS % total_cols

    # 收集所有列
    all_cols = []  # (cx0, cx1, ty0, ty1, col_global_idx)
    global_col = 0
    for tx0, tx1, ty0, ty1, n_cols in seg_list:
        print("  段 x=[%d:%d] y=[%d:%d] %d列" % (tx0, tx1, ty0, ty1, n_cols))
        cols = find_columns(binary, tx0, tx1, ty0, ty1, n_cols, ltr)
        for cx0, cx1 in cols:
            all_cols.append((cx0, cx1, ty0, ty1, global_col))
            global_col += 1

    col_chars = []
    for i in range(total_cols):
        col_chars.append(avg + (1 if i >= total_cols - remainder else 0))

    print("  总列数: %d, 字数分配: %s" % (total_cols, col_chars))

    # 预览模式
    if preview:
        preview_img = stitched.copy()
        draw = ImageDraw.Draw(preview_img)
        for i, (cx0, cx1, ty0, ty1, _) in enumerate(all_cols):
            col_height = ty1 - ty0
            draw.line([(cx0, ty0), (cx0, ty1)], fill="red", width=2)
            draw.line([(cx1, ty0), (cx1, ty1)], fill="blue", width=2)
            n_ch = col_chars[i]
            ch_h = col_height / n_ch
            for ci in range(1, n_ch):
                y = int(ty0 + ci * ch_h)
                draw.line([(cx0, y), (cx1, y)], fill="green", width=1)
        preview_path = "/tmp/%s_preview.jpg" % version_name
        preview_img.save(preview_path, quality=85)
        print("  预览已保存: %s" % preview_path)
        return 0

    # 裁切
    os.makedirs(output_dir, exist_ok=True)
    for f in os.listdir(output_dir):
        if f.endswith(".jpg") and f[:3].isdigit():
            os.remove(os.path.join(output_dir, f))

    char_idx = 0
    padding = 4
    for col_i, (cx0, cx1, ty0, ty1, _) in enumerate(all_cols):
        col_height = ty1 - ty0
        n_chars = col_chars[col_i]
        char_h = col_height / n_chars

        for ci in range(n_chars):
            char_idx += 1
            y0 = int(ty0 + ci * char_h - padding)
            y1 = int(ty0 + (ci + 1) * char_h + padding)
            x0 = max(0, cx0 - padding)
            x1 = min(w, cx1 + padding)
            y0 = max(0, y0)
            y1 = min(h, y1)

            crop_img = stitched.crop((x0, y0, x1, y1))
            crop_img = make_square(crop_img, 300)
            out_path = os.path.join(output_dir, "%03d.jpg" % char_idx)
            crop_img.save(out_path, "JPEG", quality=92)

    print("  裁切完成: %d 个字" % char_idx)
    return char_idx


def main():
    if len(sys.argv) < 2:
        print("用法: python3 tools/crop_guided.py <version> [--preview]")
        print("版本: chu, yu, dingwu")
        sys.exit(1)

    version = sys.argv[1]
    preview = "--preview" in sys.argv

    if version not in VERSIONS:
        print("未知版本: %s" % version)
        sys.exit(1)

    cfg = VERSIONS[version]
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_dir = os.path.join(base, "assets", version, "full")
    output_dir = os.path.join(base, "assets", version)

    # 加载源图
    if "source" in cfg:
        print("加载源图: %s" % cfg["source"])
        stitched = Image.open(cfg["source"])
    else:
        stitched_path = os.path.join(full_dir, "_stitched.jpg")
        if os.path.exists(stitched_path):
            print("加载已拼接图: %s" % stitched_path)
            stitched = Image.open(stitched_path)
        else:
            print("拼接分段图...")
            stitched = stitch_segments(full_dir, cfg["prefix"], cfg["nums"])
            stitched.save(stitched_path, "JPEG", quality=95)
            print("  已保存拼接图: %s (%dx%d)" % (stitched_path, *stitched.size))

    print("处理 %s..." % version)
    count = crop_version(stitched, output_dir, version, cfg, preview)
    if not preview:
        print("完成! %d 个字保存到 %s/" % (count, output_dir))


if __name__ == "__main__":
    main()
