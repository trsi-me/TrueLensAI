/* General script — active link highlight + mobile menu */
(function () {
  var path = window.location.pathname;
  document.querySelectorAll(".nav-links a").forEach(function (a) {
    if (a.getAttribute("href") === path) {
      a.classList.add("is-active");
    }
  });

  var MQ = window.matchMedia("(max-width: 900px)");
  var header = document.getElementById("site-header");
  var toggle = document.getElementById("nav-toggle");
  var nav = document.getElementById("nav-menu");

  function setMenuOpen(open) {
    if (!header || !toggle || !nav) return;
    header.classList.toggle("is-menu-open", open);
    nav.classList.toggle("is-open", open);
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    toggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    document.body.classList.toggle("nav-menu-open", open);
  }

  function closeMenu() {
    setMenuOpen(false);
  }

  if (toggle && nav && header) {
    toggle.addEventListener("click", function () {
      if (!MQ.matches) return;
      setMenuOpen(!nav.classList.contains("is-open"));
    });

    nav.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", function () {
        if (MQ.matches) closeMenu();
      });
    });

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && nav.classList.contains("is-open")) {
        closeMenu();
        toggle.focus();
      }
    });

    window.addEventListener("resize", function () {
      if (!MQ.matches) closeMenu();
    });
  }
})();
