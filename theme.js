/* 兰亭序 · 双主题切换
 *
 * dark  = 夜帖古卷（深底 + 金石色）
 * light = 兰亭春色（暖白底 + 温润色）
 *
 * 优先级：localStorage > prefers-color-scheme > dark（默认）
 * 切换时存 localStorage，下次访问保持。
 */
(function () {
  "use strict";

  var KEY = "lt-theme";
  var html = document.documentElement;

  function getSystemTheme() {
    return window.matchMedia("(prefers-color-scheme: light)").matches
      ? "light"
      : "dark";
  }

  function apply(theme) {
    html.setAttribute("data-theme", theme);
  }

  /* 初始化：localStorage → 系统 → dark */
  var saved = localStorage.getItem(KEY);
  apply(saved || getSystemTheme());

  /* 系统主题变化时，如果没有手动覆盖，跟随系统 */
  window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", function () {
    if (!localStorage.getItem(KEY)) apply(getSystemTheme());
  });

  /* 暴露 toggle 给按钮调用 */
  window.ltToggleTheme = function () {
    var current = html.getAttribute("data-theme");
    var next = current === "dark" ? "light" : "dark";
    apply(next);
    localStorage.setItem(KEY, next);
  };
})();
