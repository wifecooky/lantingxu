#!/usr/bin/env python3
"""兰亭序 · PaddleOCR 辅助单字裁切

用 PaddleOCR 检测文本列的多边形边界，
用识别文字与原文匹配筛掉题跋，
在每列内按匹配字数等分切字。

用法: python3 tools/crop_paddle.py <version> [--preview]
"""

import sys
import os
import cv2
import numpy as np

EXPECTED = 324

# 兰亭序全文（无标点）
FULL_TEXT = (
    "永和九年歲在癸丑暮春之初會于會稽山陰之蘭亭修禊事也"
    "群賢畢至少長咸集此地有崇山峻嶺茂林修竹"
    "又有清流激湍映帶左右引以為流觴曲水列坐其次"
    "雖無絲竹管弦之盛一觴一詠亦足以暢敘幽情"
    "是日也天朗氣清惠風和暢仰觀宇宙之大俯察品類之盛"
    "所以遊目騁懷足以極視聽之娛信可樂也"
    "夫人之相與俯仰一世或取諸懷抱悟言一室之內"
    "或因寄所託放浪形骸之外雖趣舍萬殊靜躁不同"
    "當其欣於所遇暫得於己快然自足不知老之將至"
    "及其所之既倦情隨事遷感慨係之矣向之所欣俯仰之間已為陳跡"
    "猶不能不以之興懷況修短隨化終期於盡"
    "古人云死生亦大矣豈不痛哉"
    "每覽昔人興感之由若合一契未嘗不臨文嗟悼不能喻之於懷"
    "固知一死生為虛誕齊彭殤為妄作"
    "後之視今亦猶今之視昔悲夫"
    "故列敘時人錄其所述雖世殊事異所以興懷其致一也"
    "後之覽者亦將有感於斯文"
)

VERSIONS = {
    "chu": {
        "path": "assets/chu/full/_stitched.jpg",
        "text_x": (0.10, 0.314), "text_y": (0.04, 0.93),
    },
    "yu": {
        "path": "assets/yu/full/_stitched.jpg",
        "text_x": (0.62, 0.925), "text_y": (0.04, 0.93),
    },
    "dingwu": {
        "path": "assets/dingwu/full/_stitched.jpg",
        "text_x": (0.04, 0.96), "text_y": (0.04, 0.96),
    },
}


def match_to_fulltext(ocr_text):
    """找 OCR 文字在原文中的最佳匹配位置和匹配字数"""
    if not ocr_text:
        return 0, -1, 0
    best_matched = 0
    best_pos = -1
    for start in range(len(FULL_TEXT)):
        matched = 0
        fi = start
        for ch in ocr_text:
            if fi < len(FULL_TEXT) and ch == FULL_TEXT[fi]:
                matched += 1
                fi += 1
            elif fi + 1 < len(FULL_TEXT) and ch == FULL_TEXT[fi + 1]:
                # 允许跳过一个字（OCR 漏识别）
                matched += 1
                fi += 2
        if matched > best_matched:
            best_matched = matched
            best_pos = start
    score = best_matched / max(len(ocr_text), 1)
    return score, best_pos, best_matched


def poly_to_rect(poly):
    """多边形 → 外接矩形 (x0, y0, x1, y1)"""
    pts = np.array(poly)
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def make_square(img, size=300):
    h, w = img.shape[:2]
    if h < 5 or w < 5:
        return np.full((size, size, 3), 200, dtype=np.uint8)
    ratio = min(size / w, size / h) * 0.85
    new_w, new_h = max(1, int(w * ratio)), max(1, int(h * ratio))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    edges = np.concatenate([resized[0], resized[-1], resized[:, 0], resized[:, -1]])
    bg = edges.mean(axis=0).astype(np.uint8)
    square = np.full((size, size, 3), bg, dtype=np.uint8)
    square[(size-new_h)//2:(size-new_h)//2+new_h,
           (size-new_w)//2:(size-new_w)//2+new_w] = resized
    return square


def process_version(ver_name, cfg, preview=False):
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    from paddleocr import PaddleOCR

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(base, cfg["path"])
    output_dir = os.path.join(base, "assets", ver_name)

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    tx = cfg["text_x"]
    ty = cfg["text_y"]
    x_off = int(w * tx[0])
    y_off = int(h * ty[0])
    text_img = img[int(h*ty[0]):int(h*ty[1]), int(w*tx[0]):int(w*tx[1])]
    th, tw = text_img.shape[:2]
    print("  正文区: %dx%d" % (tw, th))

    # 保存临时文件给 PaddleOCR
    tmp_path = "/tmp/%s_text.jpg" % ver_name
    cv2.imwrite(tmp_path, text_img)

    # OCR 检测
    ocr = PaddleOCR(lang='ch')
    result = ocr.ocr(tmp_path)
    r = result[0]
    d = dict(r)
    polys = d.get('dt_polys', [])
    texts = d.get('rec_texts', [])
    print("  OCR 检测到 %d 个区域" % len(polys))

    # 匹配每个区域到原文
    columns = []  # (score, pos, n_chars_ocr, rect, text)
    for poly, text in zip(polys, texts):
        score, pos, matched = match_to_fulltext(text)
        rect = poly_to_rect(poly)
        columns.append({
            "score": score, "pos": pos, "matched": matched,
            "n_ocr": len(text), "rect": rect, "text": text,
            "poly": poly,
        })

    # 按匹配分数筛选正文列（score > 0.3）
    text_cols = [c for c in columns if c["score"] > 0.3 and c["n_ocr"] >= 3]
    # 按原文位置排序
    text_cols.sort(key=lambda c: c["pos"])
    print("  匹配到 %d 个正文列" % len(text_cols))

    # 去重：如果两个列的 pos 接近且 rect 重叠，保留分数高的
    deduped = []
    for c in text_cols:
        overlap = False
        for d2 in deduped:
            # x 范围重叠超过 50%
            x_overlap = max(0, min(c["rect"][2], d2["rect"][2]) - max(c["rect"][0], d2["rect"][0]))
            x_union = max(c["rect"][2], d2["rect"][2]) - min(c["rect"][0], d2["rect"][0])
            if x_union > 0 and x_overlap / x_union > 0.5:
                overlap = True
                if c["score"] > d2["score"]:
                    deduped.remove(d2)
                    deduped.append(c)
                break
        if not overlap:
            deduped.append(c)
    deduped.sort(key=lambda c: c["pos"])
    print("  去重后 %d 个正文列" % len(deduped))

    # 统计识别到的总字数
    total_ocr = sum(c["n_ocr"] for c in deduped)
    print("  OCR 总字数: %d (期望 %d)" % (total_ocr, EXPECTED))

    # 预览
    if preview:
        prev = text_img.copy()
        for i, c in enumerate(deduped):
            x0, y0, x1, y1 = [int(v) for v in c["rect"]]
            cv2.rectangle(prev, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(prev, "%d(%d)" % (i+1, c["n_ocr"]),
                        (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            # 画字分割线
            ch_h = (y1 - y0) / max(c["n_ocr"], 1)
            for ci in range(1, c["n_ocr"]):
                cy = int(y0 + ci * ch_h)
                cv2.line(prev, (x0, cy), (x1, cy), (255, 0, 0), 1)
        out = "/tmp/%s_paddle_preview.jpg" % ver_name
        cv2.imwrite(out, prev)
        print("  预览: %s" % out)
        return

    # 裁切
    os.makedirs(output_dir, exist_ok=True)
    for f in os.listdir(output_dir):
        if f.endswith(".jpg") and f[:3].isdigit():
            os.remove(os.path.join(output_dir, f))

    char_idx = 0
    pad = 4
    for c in deduped:
        x0, y0, x1, y1 = [int(v) for v in c["rect"]]
        n_chars = c["n_ocr"]
        ch_h = (y1 - y0) / max(n_chars, 1)
        for ci in range(n_chars):
            char_idx += 1
            if char_idx > EXPECTED:
                break
            cy0 = max(0, int(y0 + ci * ch_h - pad))
            cy1 = min(th, int(y0 + (ci + 1) * ch_h + pad))
            cx0 = max(0, x0 - pad)
            cx1 = min(tw, x1 + pad)
            crop = text_img[cy0:cy1, cx0:cx1]
            crop = make_square(crop, 300)
            out_path = os.path.join(output_dir, "%03d.jpg" % char_idx)
            cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print("  裁切: %d 个字" % char_idx)
    return char_idx


def main():
    if len(sys.argv) < 2:
        print("用法: python3 tools/crop_paddle.py <version> [--preview]")
        print("版本: chu, yu, dingwu")
        sys.exit(1)

    ver = sys.argv[1]
    preview = "--preview" in sys.argv

    if ver not in VERSIONS:
        print("未知: %s" % ver)
        sys.exit(1)

    print("处理 %s..." % ver)
    process_version(ver, VERSIONS[ver], preview)


if __name__ == "__main__":
    main()
