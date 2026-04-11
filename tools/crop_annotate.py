#!/usr/bin/env python3
"""兰亭序 · 半自动标注裁切工具 v2

按分段处理（不拼接），每段内右→左读列。

工作流：
  python3 tools/crop_annotate.py <version> --init      自动检测 → 生成 annotation.json
  手动编辑 assets/<version>/annotation.json             校准每列字数，总和 = 324
  python3 tools/crop_annotate.py <version> --preview    预览切割线
  python3 tools/crop_annotate.py <version> --crop       执行裁切
  python3 tools/crop_annotate.py <version> --review     生成逐字审查 HTML
"""

import sys
import os
import json
import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# 兰亭序全文 324 字（无标点）
# ---------------------------------------------------------------------------
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
    "及其所之既倦情隨事遷感慨係之矣"
    "向之所欣俯仰之間已為陳跡猶不能不以之興懷"
    "況修短隨化終期於盡"
    "古人云死生亦大矣豈不痛哉"
    "每覽昔人興感之由若合一契未嘗不臨文嗟悼不能喻之於懷"
    "固知一死生為虛誕齊彭殤為妄作"
    "後之視今亦猶今之視昔悲夫"
    "故列敘時人錄其所述雖世殊事異所以興懷其致一也"
    "後之覽者亦將有感於斯文"
)

EXPECTED = 324

# 褚摹本特殊处理：崇山峻合为一格（322字）
CHU_TEXT = list(FULL_TEXT[:36]) + ["崇山峻"] + list(FULL_TEXT[39:])
CHU_EXPECTED = 322  # len(CHU_TEXT) = 322

# ---------------------------------------------------------------------------
# 每版本配置：哪些分段包含正文
# text_segments: 包含兰亭序正文的分段文件名列表
# text_region: [x0%, y0%, x1%, y1%] 正文在每个分段中的大致位置
# ---------------------------------------------------------------------------
DEFAULTS = {
    "chu": {
        "text_segments": ["csl03.jpg", "csl04.jpg", "csl05.jpg", "csl06.jpg"],
        "text_region": [0.0, 0.0, 1.0, 0.97],
    },
    "yu": {
        "text_segments": [
            "ysn01.jpg", "ysn02.jpg", "ysn03.jpg", "ysn04.jpg",
            "ysn05.jpg", "ysn06.jpg", "ysn07.jpg",
        ],
        "text_region": [0.0, 0.0, 1.0, 0.92],
    },
    "dingwu": {
        "text_segments": [
            "dw01.jpg", "dw02.jpg", "dw03.jpg", "dw04.jpg",
            "dw05.jpg", "dw06.jpg",
        ],
        "text_region": [0.0, 0.0, 1.0, 0.95],
    },
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def otsu(gray):
    hist, _ = np.histogram(gray.ravel(), 256, (0, 256))
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)
    sum_bg = w_bg = 0.0
    best_var = best_t = 0
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        var = w_bg * w_fg * (sum_bg / w_bg - (sum_all - sum_bg) / w_fg) ** 2
        if var > best_var:
            best_var = var
            best_t = t
    return best_t


def smooth(a, w=5):
    return np.convolve(a, np.ones(w) / w, "same")


def detect_columns(binary):
    """检测列边界，返回 [(x0, x1), ...] 从左到右"""
    h, w = binary.shape
    v_proj = smooth(binary.sum(axis=0), max(3, w // 50))
    threshold = v_proj.max() * 0.05
    min_width = max(8, w // 30)
    min_gap = max(3, w // 100)

    above = v_proj > threshold
    regions = []
    start = None
    for i in range(len(above)):
        if above[i] and start is None:
            start = i
        elif not above[i] and start is not None:
            if i - start >= min_width:
                regions.append((start, i))
            start = None
    if start is not None and len(above) - start >= min_width:
        regions.append((start, len(above)))

    if regions:
        merged = [regions[0]]
        for r in regions[1:]:
            if r[0] - merged[-1][1] < min_gap:
                merged[-1] = (merged[-1][0], r[1])
            else:
                merged.append(r)
        regions = merged

    return regions


def estimate_chars_in_col(binary, x0, x1):
    col = binary[:, x0:x1]
    h = col.shape[0]
    h_proj = smooth(col.sum(axis=1), max(3, h // 40))
    threshold = h_proj.max() * 0.10
    min_size = max(5, h // 25)
    min_gap = max(2, h // 80)

    above = h_proj > threshold
    regions = []
    start = None
    for i in range(len(above)):
        if above[i] and start is None:
            start = i
        elif not above[i] and start is not None:
            if i - start >= min_size:
                regions.append((start, i))
            start = None
    if start is not None and len(above) - start >= min_size:
        regions.append((start, len(above)))

    if regions:
        merged = [regions[0]]
        for r in regions[1:]:
            if r[0] - merged[-1][1] < min_gap:
                merged[-1] = (merged[-1][0], r[1])
            else:
                merged.append(r)
        regions = merged

    return max(1, len(regions))


def make_square(img, size=300):
    w, h = img.size
    ratio = min(size / w, size / h) * 0.88
    new_w, new_h = int(w * ratio), int(h * ratio)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    arr = np.array(img)
    edges = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    bg = tuple(edges.mean(axis=0).astype(int))
    sq = Image.new("RGB", (size, size), bg)
    sq.paste(img, ((size - new_w) // 2, (size - new_h) // 2))
    return sq


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

def get_base():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ann_path(version):
    return os.path.join(get_base(), "assets", version, "annotation.json")


def seg_path(version, filename):
    return os.path.join(get_base(), "assets", version, "full", filename)


def load_annotation(version):
    p = ann_path(version)
    if not os.path.exists(p):
        print("错误: %s 不存在" % p)
        print("请先运行: python3 tools/crop_annotate.py %s --init" % version)
        sys.exit(1)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# --init: 对每个正文分段自动检测列 + 估算字数
# ---------------------------------------------------------------------------

def cmd_init(version):
    cfg = DEFAULTS[version]
    tr = cfg["text_region"]

    segments = []
    total_chars = 0

    for seg_file in cfg["text_segments"]:
        path = seg_path(version, seg_file)
        if not os.path.exists(path):
            print("跳过（不存在）: %s" % seg_file)
            continue

        img = Image.open(path)
        w, h = img.size
        tx0, ty0 = int(w * tr[0]), int(h * tr[1])
        tx1, ty1 = int(w * tr[2]), int(h * tr[3])
        text_img = img.crop((tx0, ty0, tx1, ty1))

        gray = np.array(text_img.convert("L"))
        th = otsu(gray)
        binary = (gray < th).astype(np.float32)

        # 检测列
        cols = detect_columns(binary)
        # 过滤太窄的列（< 图宽的 6%，可能是印章或边框）
        tw_img = text_img.size[0]
        min_col_w = tw_img * 0.06
        cols = [(x0, x1) for x0, x1 in cols if x1 - x0 >= min_col_w]
        # 右→左阅读
        cols.reverse()

        print("%s: %dx%d, Otsu=%d, %d 列" % (seg_file, text_img.size[0], text_img.size[1], th, len(cols)))

        # 估算每列字数
        # 启发式：书法字近似方形，字高 ≈ 列宽 × 1.2（留间距）
        text_h = binary.shape[0]
        seg_cols = []
        for x0, x1 in cols:
            col_w = x1 - x0
            char_h = col_w * 1.2  # 字高略大于列宽
            n = max(1, round(text_h / char_h))
            seg_cols.append([int(x0), int(x1), n])
            total_chars += n

        segments.append({
            "file": seg_file,
            "text_region": cfg["text_region"],
            "columns": seg_cols,
        })

    annotation = {
        "_说明": "每个 segment 内 columns: [x0, x1, n_chars]，坐标相对于 text_region。右→左阅读。总字数须 = 324",
        "segments": segments,
        "_字数总和": total_chars,
    }

    out = ann_path(version)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)

    for seg in segments:
        chars = [c[2] for c in seg["columns"]]
        print("  %s: %d 列, %s (小计 %d)" % (seg["file"], len(chars), chars, sum(chars)))

    print()
    print("已保存: %s" % out)
    print("字数总和: %d (期望 %d)" % (total_chars, EXPECTED))
    if total_chars != EXPECTED:
        print("⚠ 请编辑 annotation.json 校准每列字数")
    else:
        print("✓ 总和正确")


# ---------------------------------------------------------------------------
# --preview: 每个分段画列和字分割线
# ---------------------------------------------------------------------------

def cmd_preview(version):
    ann = load_annotation(version)
    total = sum(c[2] for seg in ann["segments"] for c in seg["columns"])
    if total != EXPECTED:
        print("警告: 字数总和 = %d ≠ %d" % (total, EXPECTED))

    char_idx = 0
    previews = []

    for seg in ann["segments"]:
        path = seg_path(version, seg["file"])
        img = Image.open(path)
        w, h = img.size
        tr = seg["text_region"]
        tx0, ty0 = int(w * tr[0]), int(h * tr[1])
        tx1, ty1 = int(w * tr[2]), int(h * tr[3])
        col_h = ty1 - ty0

        draw = ImageDraw.Draw(img)

        for ci, col in enumerate(seg["columns"]):
            x0, x1, n = col[0], col[1], col[2]
            ax0, ax1 = tx0 + x0, tx0 + x1

            draw.line([(ax0, ty0), (ax0, ty1)], fill="red", width=2)
            draw.line([(ax1, ty0), (ax1, ty1)], fill="blue", width=2)

            mid_x = (ax0 + ax1) // 2
            draw.text((mid_x - 5, ty0 - 12), str(ci + 1), fill="red")

            ch_h = col_h / n
            for ci2 in range(n):
                char_idx += 1
                y = int(ty0 + ci2 * ch_h)
                draw.line([(ax0, y), (ax1, y)], fill=(0, 200, 0), width=1)
                draw.text((ax0 + 2, int(ty0 + (ci2 + 0.3) * ch_h)), str(char_idx), fill="yellow")

        previews.append(img)

    # 拼成一张预览图（水平排列）
    total_w = sum(p.width for p in previews)
    max_h = max(p.height for p in previews)
    combined = Image.new("RGB", (total_w, max_h), (40, 40, 40))
    x = 0
    for p in previews:
        combined.paste(p, (x, 0))
        x += p.width

    out = "/tmp/%s_preview.jpg" % version
    combined.save(out, quality=92)
    cw, ch2 = combined.size
    print("预览: %s (%dx%d, %d 字, %d 段)" % (out, cw, ch2, char_idx, len(ann["segments"])))


# ---------------------------------------------------------------------------
# --crop: 裁切
# ---------------------------------------------------------------------------

def cmd_crop(version):
    ann = load_annotation(version)
    total = sum(c[2] for seg in ann["segments"] for c in seg["columns"])
    if total < 300:
        print("错误: 字数总和 = %d，太少了" % total)
        sys.exit(1)
    print("总字数: %d" % total)

    output_dir = os.path.join(get_base(), "assets", version)

    # 清理旧编号文件
    for f in os.listdir(output_dir):
        if f.endswith(".jpg") and len(f) == 7 and f[:3].isdigit():
            os.remove(os.path.join(output_dir, f))

    char_idx = 0
    pad = 6

    for seg in ann["segments"]:
        path = seg_path(version, seg["file"])
        img = Image.open(path)
        w, h = img.size
        tr = seg["text_region"]
        tx0, ty0 = int(w * tr[0]), int(h * tr[1])
        tx1, ty1 = int(w * tr[2]), int(h * tr[3])
        col_h = ty1 - ty0

        for col in seg["columns"]:
            x0, x1, n = col[0], col[1], col[2]
            splits = col[3] if len(col) > 3 else None
            ax0, ax1 = tx0 + x0, tx0 + x1

            for ci in range(n):
                if splits:
                    y0_f = splits[ci - 1] if ci > 0 else 0.0
                    y1_f = splits[ci] if ci < len(splits) else 1.0
                else:
                    y0_f = ci / n
                    y1_f = (ci + 1) / n

                char_idx += 1
                cy0 = max(0, int(ty0 + y0_f * col_h - pad))
                cy1 = min(h, int(ty0 + y1_f * col_h + pad))
                cx0 = max(0, ax0 - pad)
                cx1 = min(w, ax1 + pad)

                crop = img.crop((cx0, cy0, cx1, cy1))
                crop = make_square(crop, 300)
                crop.save(os.path.join(output_dir, "%03d.jpg" % char_idx),
                          "JPEG", quality=92)

    print("裁切完成: %d 字 → %s/" % (char_idx, output_dir))


# ---------------------------------------------------------------------------
# --edit: 可拖拽的交互式标注编辑器
# ---------------------------------------------------------------------------

def cmd_edit(version):
    import io
    import base64

    ann = load_annotation(version)
    version_names = {"chu": "褚摹本", "yu": "虞摹本", "dingwu": "定武本"}
    title = version_names.get(version, version)

    # 将每个分段裁切到 text_region，编码为 base64
    seg_list = []
    for seg in ann["segments"]:
        path = seg_path(version, seg["file"])
        img = Image.open(path)
        w, h = img.size
        tr = seg["text_region"]
        tx0, ty0 = int(w * tr[0]), int(h * tr[1])
        tx1, ty1 = int(w * tr[2]), int(h * tr[3])
        text_img = img.crop((tx0, ty0, tx1, ty1))

        buf = io.BytesIO()
        text_img.save(buf, "JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()

        seg_list.append({
            "file": seg["file"],
            "text_region": seg["text_region"],
            "width": text_img.width,
            "height": text_img.height,
            "b64": b64,
            "columns": seg["columns"],
        })

    data_json = json.dumps(seg_list, ensure_ascii=False)

    # Note: This HTML is a local-only tool generated from trusted annotation
    # data. All content is from local image files and user annotations.
    # innerHTML usage is safe here as no untrusted input is involved.
    html = _build_editor_html(title, data_json)

    out = "/tmp/%s_editor.html" % version
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("编辑器: %s" % out)
    print("open %s" % out)


def _build_editor_html(title, data_json):
    """Build the interactive editor HTML. All data is from trusted local sources."""
    parts = []
    parts.append('<!DOCTYPE html>\n<html lang="zh">\n<head>\n<meta charset="utf-8">')
    parts.append('<title>' + title + ' - 标注编辑器</title>')
    parts.append('<style>')
    parts.append("""
*{box-sizing:border-box;margin:0;padding:0}
body{background:#1a1a1a;color:#ddd;font-family:system-ui,sans-serif;padding:20px}
h1{color:#e0c080;margin-bottom:10px}
.toolbar{background:#2a2a2a;padding:10px 16px;border-radius:6px;margin-bottom:20px;
         display:flex;align-items:center;gap:16px;position:sticky;top:0;z-index:100}
.total{font-size:18px;font-weight:bold}
.total.ok{color:#4c4}.total.bad{color:#c44}
.btn{background:#444;color:#ddd;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;font-size:14px}
.btn:hover{background:#555}
.btn-primary{background:#2a6}.btn-primary:hover{background:#3b7}
.segment{margin-bottom:30px}
.seg-title{color:#80b0e0;font-size:15px;margin-bottom:6px;font-weight:bold}
.seg-wrap{position:relative;display:inline-block;border:1px solid #444}
.seg-wrap img{display:block;height:700px;width:auto;user-select:none;-webkit-user-drag:none}
.col-ov{position:absolute;top:0;bottom:0;
        border-left:2px solid rgba(255,80,80,0.9);border-right:2px solid rgba(80,80,255,0.9)}
.col-ov:nth-child(odd of .col-ov){background:rgba(255,255,0,0.08)}
.col-ov:nth-child(even of .col-ov){background:rgba(0,200,255,0.08)}
.col-ov .handle{position:absolute;top:0;bottom:0;width:14px;cursor:col-resize;z-index:5}
.col-ov .handle:hover{background:rgba(255,255,255,0.25)}
.col-ov .handle-l{left:-7px}
.col-ov .handle-r{right:-7px}
.col-ov .cinput{position:absolute;top:6px;left:50%;transform:translateX(-50%);
                width:40px;height:24px;text-align:center;background:rgba(0,0,0,0.8);
                color:#e0c080;border:1px solid #666;border-radius:3px;font-size:14px;z-index:10}
.col-ov .clabel{position:absolute;bottom:6px;left:50%;transform:translateX(-50%);
                font-size:11px;color:rgba(255,255,255,0.5);z-index:10;white-space:nowrap}
.col-ov .rm-btn{position:absolute;top:34px;left:50%;transform:translateX(-50%);
                width:20px;height:20px;line-height:18px;text-align:center;
                background:rgba(200,0,0,0.7);color:#fff;border:none;border-radius:50%;
                cursor:pointer;font-size:12px;z-index:10}
.hline{position:absolute;left:0;right:0;height:15px;margin-top:-7px;
       cursor:row-resize;z-index:3;
       background:linear-gradient(transparent 6px,rgba(0,220,0,0.6) 6px,rgba(0,220,0,0.6) 9px,transparent 9px)}
.hline:hover{background:linear-gradient(transparent 4px,rgba(255,60,60,0.9) 4px,rgba(255,60,60,0.9) 11px,transparent 11px)}
.add-btn{margin-top:6px}
.help{color:#888;font-size:13px;margin-bottom:16px}
#jout{width:100%;height:250px;background:#222;color:#aaa;border:1px solid #555;
      font-family:monospace;font-size:12px;padding:10px;margin-top:10px;display:none}
""")
    parts.append('</style>\n</head>\n<body>')
    parts.append('<h1>' + title + ' - 标注编辑器</h1>')
    parts.append('<p class="help">拖拽红/蓝边界调列位置 | 双击空白加横线 | Shift+点击横线删除 | 拖拽横线移动 | 横线悬停变红</p>')
    parts.append("""
<div class="toolbar">
  <span class="total" id="total">总字数: 0 / 324</span>
  <button class="btn btn-primary" onclick="downloadJSON()">下载 annotation.json</button>
  <button class="btn" onclick="showJSON()">显示 JSON</button>
</div>
<div id="segs"></div>
<textarea id="jout" readonly></textarea>
""")
    parts.append('<script>')
    parts.append('const EXPECTED=324;')
    parts.append('const segData=' + data_json + ';')
    parts.append(r"""
let state=segData.map(s=>({
  file:s.file,text_region:s.text_region,width:s.width,height:s.height,
  columns:s.columns.map(c=>[...c])
}));

function render(){
  const ct=document.getElementById('segs');
  while(ct.firstChild)ct.removeChild(ct.firstChild);
  let gi=0;
  for(let si=0;si<state.length;si++){
    const seg=state[si];const sd=segData[si];
    const dv=document.createElement('div');dv.className='segment';
    const tt=document.createElement('div');tt.className='seg-title';
    const st=seg.columns.reduce((s,c)=>s+c[2],0);
    tt.textContent=seg.file+' ('+seg.columns.length+' 列, '+st+' 字, #'+(gi+1)+'-#'+(gi+st)+')';
    dv.appendChild(tt);
    const wr=document.createElement('div');wr.className='seg-wrap';wr.id='w'+si;
    const im=document.createElement('img');
    im.src='data:image/jpeg;base64,'+sd.b64;im.draggable=false;
    wr.appendChild(im);
    for(let ci=0;ci<seg.columns.length;ci++){
      const[x0,x1,n]=seg.columns[ci];
      const ov=document.createElement('div');ov.className='col-ov';
      ov.style.left=(x0/seg.width*100)+'%';
      ov.style.width=((x1-x0)/seg.width*100)+'%';
      const inp=document.createElement('input');inp.className='cinput';
      inp.type='number';inp.min=1;inp.max=30;inp.value=n;
      inp.onchange=function(){
        const nv=parseInt(this.value)||1;seg.columns[ci][2]=nv;
        const sp=[];for(let i=1;i<nv;i++)sp.push(i/nv);
        seg.columns[ci][3]=sp;render();
      };
      ov.appendChild(inp);
      const lb=document.createElement('div');lb.className='clabel';
      lb.textContent='#'+(gi+1)+'-#'+(gi+n);ov.appendChild(lb);
      if(seg.columns.length>1){
        const rm=document.createElement('button');rm.className='rm-btn';rm.textContent='\u00d7';
        rm.title='delete';rm.onclick=function(e){e.stopPropagation();seg.columns.splice(ci,1);render()};
        ov.appendChild(rm);
      }
      // splits: N-1 fractional positions for division lines
      if(!seg.columns[ci][3]){
        const sp=[];for(let i=1;i<n;i++)sp.push(i/n);
        seg.columns[ci][3]=sp;
      }
      const sp=seg.columns[ci][3];
      for(let i=0;i<sp.length;i++){
        const ln=document.createElement('div');ln.className='hline';
        ln.style.top=(sp[i]*100)+'%';
        ln.onmousedown=(function(ii){return function(e){
          if(e.shiftKey){e.preventDefault();e.stopPropagation();
            sp.splice(ii,1);seg.columns[ci][2]--;seg.columns[ci][3]=sp;render();return;}
          startHDrag(e,si,ci,ii);
        }})(i);
        ln.title='拖拽移动 | Shift+点击删除';
        ov.appendChild(ln);
      }
      // 双击列空白处：插入新横线
      ov.ondblclick=function(e){
        const rect=ov.getBoundingClientRect();
        const yFrac=(e.clientY-rect.top)/rect.height;
        if(yFrac<0.01||yFrac>0.99)return;
        sp.push(yFrac);sp.sort((a,b)=>a-b);
        seg.columns[ci][2]++;seg.columns[ci][3]=sp;render();
      };
      // 每格显示编号+期望文字
      const fullText="永和九年歲在癸丑暮春之初會于會稽山陰之蘭亭修禊事也群賢畢至少長咸集此地有崇山峻嶺茂林修竹又有清流激湍映帶左右引以為流觴曲水列坐其次雖無絲竹管弦之盛一觴一詠亦足以暢敘幽情是日也天朗氣清惠風和暢仰觀宇宙之大俯察品類之盛所以遊目騁懷足以極視聽之娛信可樂也夫人之相與俯仰一世或取諸懷抱悟言一室之內或因寄所託放浪形骸之外雖趣舍萬殊靜躁不同當其欣於所遇暫得於己快然自足不知老之將至及其所之既倦情隨事遷感慨係之矣向之所欣俯仰之間已為陳跡猶不能不以之興懷況修短隨化終期於盡古人云死生亦大矣豈不痛哉每覽昔人興感之由若合一契未嘗不臨文嗟悼不能喻之於懷固知一死生為虛誕齊彭殤為妄作後之視今亦猶今之視昔悲夫故列敘時人錄其所述雖世殊事異所以興懷其致一也後之覽者亦將有感於斯文";
      // 褚摹本：崇山峻合为一格（用数组，一个元素 = 一格）
      const dispChars=[];
      for(let i=0;i<36;i++) dispChars.push(fullText[i]);
      dispChars.push("崇山峻");
      for(let i=39;i<fullText.length;i++) dispChars.push(fullText[i]);
      for(let k=0;k<n;k++){
        const y0=k===0?0:sp[k-1];
        const y1=k<sp.length?sp[k]:1;
        const mid=((y0+y1)/2*100);
        const charIdx=gi+k+1;
        const ch=charIdx<=dispChars.length?dispChars[charIdx-1]:'?';
        const tag=document.createElement('div');
        tag.style.cssText='position:absolute;left:50%;top:'+mid+'%;transform:translate(-50%,-50%);font-size:10px;color:rgba(255,255,255,0.7);z-index:2;pointer-events:none;text-align:center;line-height:1.2';
        tag.textContent=charIdx+' '+ch;
        ov.appendChild(tag);
      }
      const lh=document.createElement('div');lh.className='handle handle-l';
      lh.onmousedown=function(e){startDrag(e,si,ci,'l')};ov.appendChild(lh);
      const rh=document.createElement('div');rh.className='handle handle-r';
      rh.onmousedown=function(e){startDrag(e,si,ci,'r')};ov.appendChild(rh);
      wr.appendChild(ov);
      gi+=n;
    }
    dv.appendChild(wr);
    const ab=document.createElement('button');ab.className='btn add-btn';ab.textContent='+ 添加列';
    ab.onclick=function(){
      const sorted=[...seg.columns].sort((a,b)=>a[0]-b[0]);
      let best=0,pos=seg.width/2,prev=0;
      for(const c of sorted){if(c[0]-prev>best){best=c[0]-prev;pos=prev+(c[0]-prev)/2}prev=c[1]}
      if(seg.width-prev>best)pos=prev+(seg.width-prev)/2;
      const cw=Math.min(80,best/2||40);
      seg.columns.push([Math.round(pos-cw),Math.round(pos+cw),12]);
      seg.columns.sort((a,b)=>b[0]-a[0]);render();
    };
    dv.appendChild(ab);
    ct.appendChild(dv);
  }
  updTotal();
}

function startDrag(e,si,ci,side){
  e.preventDefault();e.stopPropagation();
  const wr=document.getElementById('w'+si);
  const rect=wr.getBoundingClientRect();
  const seg=state[si];const scale=seg.width/rect.width;
  const sx=e.clientX;const sv=side==='l'?seg.columns[ci][0]:seg.columns[ci][1];
  const ovs=wr.querySelectorAll('.col-ov');const ov=ovs[ci];
  function mv(e){
    const nv=Math.round(sv+(e.clientX-sx)*scale);
    if(side==='l')seg.columns[ci][0]=Math.max(0,Math.min(nv,seg.columns[ci][1]-10));
    else seg.columns[ci][1]=Math.max(seg.columns[ci][0]+10,Math.min(nv,seg.width));
    const[x0,x1]=seg.columns[ci];
    ov.style.left=(x0/seg.width*100)+'%';ov.style.width=((x1-x0)/seg.width*100)+'%';
  }
  function up(){document.removeEventListener('mousemove',mv);
    document.removeEventListener('mouseup',up);render()}
  document.addEventListener('mousemove',mv);document.addEventListener('mouseup',up);
}

function startHDrag(e,si,ci,li){
  e.preventDefault();e.stopPropagation();
  const wr=document.getElementById('w'+si);
  const rect=wr.getBoundingClientRect();
  const seg=state[si];const sp=seg.columns[ci][3];
  const sy=e.clientY;const sv=sp[li];
  const ovs=wr.querySelectorAll('.col-ov');const ov=ovs[ci];
  const lines=ov.querySelectorAll('.hline');const ln=lines[li];
  function mv(e){
    const dy=(e.clientY-sy)/rect.height;
    let nv=sv+dy;
    const mn=li>0?sp[li-1]+0.015:0.015;
    const mx=li<sp.length-1?sp[li+1]-0.015:0.985;
    nv=Math.max(mn,Math.min(nv,mx));
    sp[li]=nv;ln.style.top=(nv*100)+'%';
  }
  function up(){document.removeEventListener('mousemove',mv);
    document.removeEventListener('mouseup',up);render()}
  document.addEventListener('mousemove',mv);document.addEventListener('mouseup',up);
}

function updTotal(){
  let t=0;for(const s of state)t+=s.columns.reduce((a,c)=>a+c[2],0);
  const el=document.getElementById('total');
  el.textContent='总字数: '+t+' / '+EXPECTED;
  el.className='total '+(t===322?'ok':'bad');
}

function getAnn(){
  return {
    "_说明":"columns: [x0, x1, n_chars]，坐标相对于 text_region。右到左阅读。总字数须 = 324",
    segments:state.map(s=>({file:s.file,text_region:s.text_region,
      columns:s.columns.map(c=>{
        const r=[c[0],c[1],c[2]];
        if(c[3])r.push(c[3].map(v=>Math.round(v*10000)/10000));
        return r;
      })})),
    "_字数总和":state.reduce((t,s)=>t+s.columns.reduce((a,c)=>a+c[2],0),0)
  };
}

function showJSON(){
  const ta=document.getElementById('jout');
  ta.style.display='block';ta.value=JSON.stringify(getAnn(),null,2);
  ta.select();ta.scrollIntoView();
}

function downloadJSON(){
  const j=JSON.stringify(getAnn(),null,2);
  const b=new Blob([j],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);
  a.download='annotation.json';a.click();URL.revokeObjectURL(a.href);
}

render();
""")
    parts.append('</script>\n</body>\n</html>')
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# --review: 审查 HTML
# ---------------------------------------------------------------------------

def cmd_review(version):
    base = get_base()
    asset_dir = os.path.join(base, "assets", version)

    chars = sorted(
        f for f in os.listdir(asset_dir)
        if f.endswith(".jpg") and len(f) == 7 and f[:3].isdigit()
    )
    if not chars:
        print("未找到裁切结果，请先运行 --crop")
        return

    ann = load_annotation(version)
    # char_idx → (seg_name, col_number)
    col_map = {}
    idx = 0
    for seg in ann["segments"]:
        for ci, col in enumerate(seg["columns"]):
            for _ in range(col[2]):
                idx += 1
                col_map[idx] = (seg["file"], ci + 1)

    version_names = {"chu": "褚摹本", "yu": "虞摹本", "dingwu": "定武本"}
    title = version_names.get(version, version)

    html = []
    html.append('<!DOCTYPE html>')
    html.append('<html lang="zh"><head><meta charset="utf-8">')
    html.append('<title>%s 逐字审查 (%d字)</title>' % (title, len(chars)))
    html.append('<style>')
    html.append("""
body { font-family: "Noto Serif SC", serif; background: #1a1a1a; color: #ddd; margin: 20px; }
h1 { color: #e0c080; }
.info { color: #999; margin-bottom: 20px; }
.seg-header { background: #2a2a3a; color: #80b0e0; padding: 10px 16px; margin: 20px 0 4px;
              border-radius: 4px; font-size: 15px; font-weight: bold; }
.col-header { background: #333; color: #e0c080; padding: 6px 16px; margin: 8px 0 4px;
              border-radius: 4px; font-size: 13px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, 130px); gap: 6px; margin-bottom: 8px; }
.cell { text-align: center; background: #222; border-radius: 4px; padding: 6px;
        transition: transform 0.15s; cursor: default; }
.cell:hover { transform: scale(1.5); z-index: 10; position: relative; }
.cell img { width: 108px; height: 108px; object-fit: contain; border-radius: 2px; }
.cell .idx { font-size: 11px; color: #666; margin-top: 2px; }
.cell .expected { font-size: 28px; color: #e0c080; margin-top: 2px; }
""")
    html.append('</style></head><body>')
    html.append('<h1>%s · 逐字审查</h1>' % title)
    html.append('<p class="info">%d / %d 字。悬停放大。检查每个裁切图与下方期望文字是否对应。</p>' % (len(chars), EXPECTED))

    current_seg = None
    current_col = None
    col_chars_html = []

    def flush():
        if col_chars_html:
            html.append('<div class="grid">')
            html.extend(col_chars_html)
            html.append('</div>')
            col_chars_html.clear()

    for i, f in enumerate(chars):
        num = i + 1
        # 褚摹本：崇山峻合并为一格
        if version == "chu":
            display_text = CHU_TEXT
        else:
            display_text = FULL_TEXT
        ch = display_text[i] if i < len(display_text) else "?"
        img_path = os.path.abspath(os.path.join(asset_dir, f))
        seg_name, col_num = col_map.get(num, ("?", 0))

        if seg_name != current_seg:
            flush()
            current_seg = seg_name
            current_col = None
            html.append('<div class="seg-header">%s</div>' % seg_name)

        if col_num != current_col:
            flush()
            current_col = col_num
            html.append('<div class="col-header">第 %d 列</div>' % col_num)

        col_chars_html.append('<div class="cell">')
        col_chars_html.append('  <img src="file://%s">' % img_path)
        col_chars_html.append('  <div class="idx">#%d</div>' % num)
        col_chars_html.append('  <div class="expected">%s</div>' % ch)
        col_chars_html.append('</div>')

    flush()
    html.append('</body></html>')

    out = "/tmp/%s_review.html" % version
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print("审查页: %s (%d 字)" % (out, len(chars)))
    print("open %s" % out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print("用法: python3 tools/crop_annotate.py <version> <command>")
        print()
        print("版本: chu, yu, dingwu")
        print()
        print("命令:")
        print("  --init      自动检测 → 生成 annotation.json")
        print("  --preview   预览切割线")
        print("  --crop      按标注裁切 324 字")
        print("  --review    生成逐字审查 HTML")
        sys.exit(1)

    ver = sys.argv[1]
    cmd = sys.argv[2]

    if ver not in DEFAULTS:
        print("未知版本: %s (支持: %s)" % (ver, ", ".join(DEFAULTS)))
        sys.exit(1)

    assert len(FULL_TEXT) == EXPECTED, "FULL_TEXT 长度 %d ≠ %d" % (len(FULL_TEXT), EXPECTED)

    cmds = {
        "--init": cmd_init, "--preview": cmd_preview,
        "--edit": cmd_edit, "--crop": cmd_crop, "--review": cmd_review,
    }
    if cmd not in cmds:
        print("未知命令: %s" % cmd)
        sys.exit(1)
    cmds[cmd](ver)


if __name__ == "__main__":
    main()
