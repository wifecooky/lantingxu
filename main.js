/* 兰亭序 · 逐字对比
 *
 * 读取 window.LT_CONFIG（在本脚本之前由 data/lanting-data.js 设置），
 * 渲染竖排全文到 #lt-scroll，每个非标点字可点击。
 * 点击后在 #lt-panel 中并列展示各版本的单字图。
 *
 * 数据结构原则：
 *   - 句子是渲染单位（一句一列），字只是位置
 *   - 标点不可点击，不分配序号，不对应单字图
 *   - 字的全局序号在渲染时动态计算（遍历句子、跳过标点、自增）
 */
(function () {
  "use strict";

  var config = window.LT_CONFIG;
  if (!config || !Array.isArray(config.sentences)) {
    console.error("[lt] window.LT_CONFIG 未设置或 sentences 缺失");
    return;
  }

  var SENTENCES = config.sentences;
  var VERSIONS = config.versions || [];
  var PUNC = new Set(config.punctuation || "");

  /* 小工具 */
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function empty(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  /* ---------- 渲染卷轴 ---------- */
  var scroll = document.getElementById("lt-scroll");
  var panelRoot = document.getElementById("lt-panel");
  if (!scroll || !panelRoot) return;

  var charIndex = 0;   /* 全局字序号（跳过标点） */
  var totalChars = 0;  /* 总字数 */
  var allChars = [];    /* 所有可点击字的 DOM 引用，按序号排列 */

  /* 先算总字数 */
  SENTENCES.forEach(function (s) {
    for (var i = 0; i < s.t.length; i++) {
      if (!PUNC.has(s.t[i])) totalChars++;
    }
  });

  /* 渲染 */
  SENTENCES.forEach(function (s, si) {
    var col = el("section", "lt-col");
    col.dataset.idx = String(si);

    for (var i = 0; i < s.t.length; i++) {
      var ch = s.t[i];
      var span = el("span", null, ch);

      if (PUNC.has(ch)) {
        span.className = "lt-pun";
        span.setAttribute("aria-hidden", "true");
      } else {
        span.className = "lt-ch";
        span.dataset.ch = ch;
        span.dataset.gidx = String(charIndex); /* global index */
        span.tabIndex = 0;
        span.setAttribute("role", "button");
        span.setAttribute("aria-label", "字 " + ch);
        allChars[charIndex] = span;
        charIndex++;
      }

      col.appendChild(span);
    }

    scroll.appendChild(col);
  });

  /* ---------- 交互：面板 ---------- */
  var activeChar = null;

  function closePanel() {
    if (activeChar) activeChar.classList.remove("lt-ch--active");
    activeChar = null;
    panelRoot.classList.remove("lt-panel--open");
    empty(panelRoot);
  }

  function padNum(n) {
    return String(n).padStart(3, "0");
  }

  function fillPanel(charEl) {
    var gidx = Number(charEl.dataset.gidx);
    var ch = charEl.dataset.ch;

    if (activeChar) activeChar.classList.remove("lt-ch--active");
    empty(panelRoot);

    /* Header */
    var head = el("div", "lt-panel__head");
    head.appendChild(el("span", "lt-panel__char", ch));
    head.appendChild(el("span", "lt-panel__pos", "第 " + (gidx + 1) + " 字 / 共 " + totalChars + " 字"));
    var closeBtn = el("button", "lt-panel__close", "×");
    closeBtn.type = "button";
    closeBtn.setAttribute("aria-label", "关闭");
    head.appendChild(closeBtn);
    panelRoot.appendChild(head);

    /* 版本并列 */
    var grid = el("div", "lt-panel__grid");

    VERSIONS.forEach(function (v) {
      var item = el("div", "lt-panel__item");

      var imgWrap = el("div", "lt-panel__img-wrap");
      var img = document.createElement("img");
      img.className = "lt-panel__img";
      img.src = "/assets/" + v.id + "/" + padNum(gidx + 1) + ".jpg";
      img.alt = v.name + " · " + ch;
      img.loading = "lazy";
      /* 加载失败 → 显示「缺」 */
      img.onerror = function () {
        imgWrap.classList.add("lt-panel__img-wrap--missing");
        img.style.display = "none";
        imgWrap.appendChild(el("span", "lt-panel__missing", "缺"));
      };
      imgWrap.appendChild(img);
      item.appendChild(imgWrap);

      item.appendChild(el("div", "lt-panel__ver-name", v.name));
      grid.appendChild(item);
    });

    panelRoot.appendChild(grid);

    charEl.classList.add("lt-ch--active");
    activeChar = charEl;
    panelRoot.classList.add("lt-panel--open");
  }

  /* 委托点击 */
  scroll.addEventListener("click", function (e) {
    var ch = e.target.closest(".lt-ch");
    if (!ch) return;
    if (activeChar === ch) { closePanel(); return; }
    fillPanel(ch);
  });

  /* 面板关闭按钮 */
  panelRoot.addEventListener("click", function (e) {
    if (e.target.closest(".lt-panel__close")) closePanel();
  });

  /* 面板外点击关闭 */
  document.addEventListener("click", function (e) {
    if (!panelRoot.classList.contains("lt-panel--open")) return;
    if (e.target.closest("#lt-panel") || e.target.closest("#lt-scroll")) return;
    closePanel();
  });

  /* 键盘：Enter/Space 激活字 */
  scroll.addEventListener("keydown", function (e) {
    if (e.key !== "Enter" && e.key !== " ") return;
    var ch = e.target.closest(".lt-ch");
    if (!ch) return;
    e.preventDefault();
    ch.click();
  });

  /* 键盘：Escape 关闭 */
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closePanel();
  });

  /* 键盘：← → 切换前/后一个字 */
  document.addEventListener("keydown", function (e) {
    if (!panelRoot.classList.contains("lt-panel--open")) return;
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    if (!activeChar) return;
    var gidx = Number(activeChar.dataset.gidx);
    /* vertical-rl: ArrowLeft = 下一字（gidx+1），ArrowRight = 上一字（gidx-1）
       但面板是水平的，所以用直觉映射：← = 前一字，→ = 后一字 */
    var next = e.key === "ArrowLeft" ? gidx - 1 : gidx + 1;
    if (next < 0 || next >= totalChars) return;
    e.preventDefault();
    fillPanel(allChars[next]);
    /* 确保该字在可视区 */
    allChars[next].scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
  });

  /* 初始：滚到最右（竖排卷轴的开头） */
  requestAnimationFrame(function () {
    scroll.scrollLeft = scroll.scrollWidth;
  });
})();
