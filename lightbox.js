/* 兰亭序 · lightbox
 *
 * 全局委派：点击 .lt-panel__img → 浮现大图。
 * ESC 或背景点击关闭。不开新页面。
 */
(function () {
  "use strict";

  if (window.__ltLightbox) return;
  window.__ltLightbox = true;

  var overlay = document.createElement("div");
  overlay.className = "lightbox";
  overlay.setAttribute("role", "dialog");
  overlay.setAttribute("aria-label", "图片预览");

  var img = document.createElement("img");
  img.className = "lightbox__img";
  img.alt = "";
  overlay.appendChild(img);

  var closeBtn = document.createElement("button");
  closeBtn.className = "lightbox__close";
  closeBtn.type = "button";
  closeBtn.setAttribute("aria-label", "关闭");
  closeBtn.textContent = "×";
  overlay.appendChild(closeBtn);

  var caption = document.createElement("div");
  caption.className = "lightbox__caption";
  overlay.appendChild(caption);

  document.body.appendChild(overlay);

  function open(src, alt) {
    img.src = src;
    img.alt = alt || "";
    caption.textContent = alt || "";
    overlay.classList.add("lightbox--open");
    document.body.style.overflow = "hidden";
  }

  function close() {
    overlay.classList.remove("lightbox--open");
    document.body.style.overflow = "";
    setTimeout(function () { img.src = ""; }, 300);
  }

  /* 委派点击 */
  document.addEventListener("click", function (e) {
    var panelImg = e.target.closest(".lt-panel__img");
    if (panelImg) {
      e.preventDefault();
      e.stopPropagation();
      open(panelImg.src, panelImg.alt);
      return;
    }
  }, true);

  /* 关闭 */
  overlay.addEventListener("click", function (e) {
    if (e.target === overlay || e.target.closest(".lightbox__close")) {
      close();
    }
  });

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && overlay.classList.contains("lightbox--open")) {
      close();
    }
  });
})();
