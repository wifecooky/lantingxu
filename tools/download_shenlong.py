#!/usr/bin/env python3
"""神龙本兰亭序单字图下载 — 组合策略

1. 从 GitHub (herrkaefer/chinese-calligraphy-vectorization) 下载已有的裁切图
2. 从 yac8.com 补齐缺失的字
"""

import os
import re
import time
import json
import base64
import requests
from urllib.parse import urljoin

EXPECTED = 324
GITHUB_API = "https://api.github.com/repos/herrkaefer/chinese-calligraphy-vectorization/contents/calligraphy/Lantingxu/original-crop"
GITHUB_RAW = "https://raw.githubusercontent.com/herrkaefer/chinese-calligraphy-vectorization/master/calligraphy/Lantingxu/original-crop/"


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "assets", "shenlong")
    os.makedirs(output_dir, exist_ok=True)

    # 清理旧编号文件
    for f in os.listdir(output_dir):
        if f.endswith(".jpg") and f[:3].isdigit():
            os.remove(os.path.join(output_dir, f))

    # === 第一步：GitHub 下载 ===
    print("=== 第一步：从 GitHub 下载 ===\n")

    # 获取文件列表
    resp = requests.get(GITHUB_API, timeout=30)
    resp.raise_for_status()
    files = resp.json()

    # 解析文件名中的编号（如 "100之.jpg" -> 100, "175.jpg" -> 175）
    github_map = {}  # num -> download_url
    for f in files:
        name = f["name"]
        if name == "冯承素摹本.jpg":
            continue
        match = re.match(r"^(\d+)", name)
        if match:
            num = int(match.group(1))
            if 1 <= num <= EXPECTED:
                github_map[num] = f["download_url"]

    print("GitHub 上有 %d 张（编号范围 %d-%d）\n" % (
        len(github_map),
        min(github_map.keys()) if github_map else 0,
        max(github_map.keys()) if github_map else 0,
    ))

    # 下载 GitHub 图片
    gh_downloaded = 0
    for num in sorted(github_map.keys()):
        out_path = os.path.join(output_dir, "%03d.jpg" % num)
        for attempt in range(3):
            try:
                r = requests.get(github_map[num], timeout=30)
                r.raise_for_status()
                with open(out_path, "wb") as fp:
                    fp.write(r.content)
                gh_downloaded += 1
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                else:
                    print("  GitHub %03d 失败: %s" % (num, e))
        if gh_downloaded % 30 == 0:
            print("  GitHub 已下载 %d..." % gh_downloaded)

    print("\nGitHub 下载完成: %d 张\n" % gh_downloaded)

    # === 第二步：找出缺失的编号 ===
    missing = []
    for n in range(1, EXPECTED + 1):
        path = os.path.join(output_dir, "%03d.jpg" % n)
        if not os.path.exists(path):
            missing.append(n)

    if not missing:
        print("全部 %d 张已就位！" % EXPECTED)
        return

    print("缺失 %d 张: %s\n" % (len(missing), missing[:20]))

    # === 第三步：yac8 补缺 ===
    print("=== 第二步：从 yac8.com 补缺 ===\n")

    # 先收集所有 yac8 图片 URL
    print("收集 yac8 图片 URL...")
    yac8_urls = collect_yac8_urls()

    if not yac8_urls:
        print("yac8 URL 收集失败")
        return

    # yac8_urls[0] 是全文图，yac8_urls[1] 对应字 1，以此类推
    char_urls = yac8_urls[1:]  # 跳过全文图
    print("yac8 收集到 %d 张字帖 URL\n" % len(char_urls))

    # 下载缺失的
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": "http://www.yac8.com/news/10725.html",
    })

    yac8_ok = 0
    yac8_fail = []
    for num in missing:
        if num - 1 >= len(char_urls):
            yac8_fail.append(num)
            continue
        src = char_urls[num - 1]
        out_path = os.path.join(output_dir, "%03d.jpg" % num)
        ok = False
        for attempt in range(3):
            try:
                r = session.get(src, timeout=30)
                r.raise_for_status()
                if r.content[:2] == b'\xff\xd8':
                    with open(out_path, "wb") as fp:
                        fp.write(r.content)
                    yac8_ok += 1
                    ok = True
                    break
                else:
                    time.sleep(2)
            except:
                time.sleep(2)
        if not ok:
            yac8_fail.append(num)

    print("yac8 补充: %d 张成功" % yac8_ok)

    # === 第四步：浏览器补缺（如果还有失败的）===
    if yac8_fail:
        print("\n=== 第三步：浏览器补缺 %d 张 ===" % len(yac8_fail))
        browser_fill(yac8_fail, char_urls, output_dir)

    # 最终统计
    final_count = sum(1 for n in range(1, EXPECTED + 1)
                      if os.path.exists(os.path.join(output_dir, "%03d.jpg" % n)))
    print("\n=== 最终结果：%d/%d 张 ===" % (final_count, EXPECTED))
    still_missing = [n for n in range(1, EXPECTED + 1)
                     if not os.path.exists(os.path.join(output_dir, "%03d.jpg" % n))]
    if still_missing:
        print("仍缺: %s" % still_missing)


def collect_yac8_urls():
    """用 requests 收集 yac8 所有图片 URL"""
    from bs4 import BeautifulSoup
    all_urls = []
    for i in range(65):
        if i:
            url = "http://www.yac8.com/news/10725_%d.html" % (i + 1)
        else:
            url = "http://www.yac8.com/news/10725.html"
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=30,
                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                content = soup.find(id="newsContent")
                if content:
                    for img in content.find_all("img"):
                        src = img.get("src")
                        if src:
                            if not src.startswith("http"):
                                src = urljoin(url, src)
                            all_urls.append(src)
                    break
            except:
                time.sleep(2)
    return all_urls


def browser_fill(missing_nums, char_urls, output_dir):
    """用 Playwright 浏览器补下载"""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright 未安装，跳过")
        return

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        )
        page = ctx.new_page()

        # 先访问首页建立 cookie
        page.goto("http://www.yac8.com/news/10725.html", wait_until="networkidle", timeout=30000)
        time.sleep(2)

        ok = 0
        for num in missing_nums:
            if num - 1 >= len(char_urls):
                continue
            src = char_urls[num - 1]
            out_path = os.path.join(output_dir, "%03d.jpg" % num)
            try:
                b64 = page.evaluate("""(url) => {
                    return fetch(url)
                        .then(r => r.blob())
                        .then(blob => new Promise((resolve, reject) => {
                            const reader = new FileReader();
                            reader.onload = () => resolve(reader.result.split(',')[1]);
                            reader.onerror = reject;
                            reader.readAsDataURL(blob);
                        }));
                }""", src)
                if b64:
                    data = base64.b64decode(b64)
                    if data[:2] == b'\xff\xd8':
                        with open(out_path, "wb") as fp:
                            fp.write(data)
                        ok += 1
            except Exception as e:
                print("  浏览器 %03d 失败: %s" % (num, str(e)[:60]))
            time.sleep(0.5)

        browser.close()
        print("浏览器补充: %d 张" % ok)


if __name__ == "__main__":
    main()
