// "Titanic Voyage" interactions.
//
// 1. The opening stage plays the custom Lottie ship scene with a gentle parallax
//    and a pause/resume control.
// 2. "Board the ship" plays a full-screen sailing transition before opening the
//    predictor.
// 3. A prediction result shows a compact voyage vignette whose tint and speed
//    reflect the outcome and the survival probability.

(function () {
  "use strict";

  var lottie = window.lottie;
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var DEFAULT_SRC = "/static/lottie/titanic.json";
  var stage = document.querySelector("[data-voyage]");
  var lottieSrc = (stage && stage.getAttribute("data-voyage-src")) || DEFAULT_SRC;

  function load(container, source, settings) {
    if (!lottie || !container) return null;
    try {
      var anim = lottie.loadAnimation({
        container: container,
        renderer: "svg",
        loop: true,
        autoplay: !reduceMotion,
        path: source,
        rendererSettings: {
          preserveAspectRatio: "xMidYMid slice",
          progressiveLoad: true
        }
      });
      return anim || null;
    } catch (error) {
      return null;
    }
  }

  function markFallback(scope) {
    if (scope) scope.classList.add("is-fallback");
  }

  // ---------------------------------------------------------------- the stage
  if (stage) {
    var art = stage.querySelector("[data-voyage-canvas]");
    var anim = load(art, lottieSrc);

    if (!anim) {
      markFallback(stage);
    } else {
      anim.addEventListener("data_failed", function () { markFallback(stage); });
      stage.classList.add("is-live");
      anim.setSpeed(reduceMotion ? 0.6 : 0.85);

      if (!reduceMotion) {
        var frame = null;
        stage.addEventListener("pointermove", function (event) {
          if (frame) return;
          frame = window.requestAnimationFrame(function () {
            frame = null;
            var box = stage.getBoundingClientRect();
            var dx = (event.clientX - box.left) / box.width - 0.5;
            var dy = (event.clientY - box.top) / box.height - 0.5;
            art.style.transform =
              "translate3d(" + (dx * -16).toFixed(2) + "px," +
              (dy * -10).toFixed(2) + "px,0) scale(1.03)";
          });
        });
        stage.addEventListener("pointerleave", function () {
          art.style.transform = "";
        });
        stage.addEventListener("pointerenter", function () {
          if (!anim.isPaused) anim.setSpeed(1.2);
        });
        stage.addEventListener("pointerleave", function () {
          if (!anim.isPaused) anim.setSpeed(0.85);
        });
      }
    }

    var toggle = stage.querySelector("[data-voyage-toggle]");
    var toggleLabel = stage.querySelector("[data-voyage-toggle-label]");
    if (toggle && anim) {
      toggle.addEventListener("click", function () {
        if (anim.isPaused) {
          anim.play();
          toggle.setAttribute("aria-pressed", "false");
          if (toggleLabel) toggleLabel.textContent = "Pause voyage";
          stage.classList.remove("is-paused");
        } else {
          anim.pause();
          toggle.setAttribute("aria-pressed", "true");
          if (toggleLabel) toggleLabel.textContent = "Resume voyage";
          stage.classList.add("is-paused");
        }
        toggle.blur();
      });
    } else if (toggle) {
      toggle.hidden = true;
    }
  }

  // ------------------------------------------------------- sailing transition
  //
  // The destination page is prefetched ahead of time (on hover/focus and once
  // the page goes idle).  That warms the serverless function and the model, so
  // the browser can paint /predict immediately instead of leaving a blank
  // blue wipe on screen while the function cold-starts.
  var WARMED = Object.create(null);

  function warm(target) {
    if (!target) return Promise.resolve();
    if (WARMED[target]) return WARMED[target];
    WARMED[target] = fetch(target, { credentials: "same-origin" })
      .then(function (response) {
        return response && response.text();
      })
      .catch(function () {
        return null;
      });
    return WARMED[target];
  }

  // Timeline: sail for 1.6 s, wipe over 0.82 s, then leave.  The wipe holds a
  // "preparing boarding pass" loader so a slow first paint never looks dead.
  var SAIL_WIPE_AT = 1600;
  var SAIL_WIPE_MS = 820;
  var SAIL_MIN_NAV = SAIL_WIPE_AT + SAIL_WIPE_MS;

  function sailTo(target) {
    var warmup = warm(target);

    if (reduceMotion) {
      window.location.href = target;
      return;
    }

    var overlay = document.createElement("div");
    overlay.className = "sail-overlay";
    overlay.innerHTML =
      '<div class="sail-scene"><div class="sail-art" data-sail-art></div></div>' +
      '<div class="sail-speed" aria-hidden="true">' +
      "<span></span><span></span><span></span><span></span><span></span>" +
      "<span></span><span></span><span></span><span></span>" +
      "</div>" +
      '<p class="sail-caption">Full steam ahead</p>' +
      '<div class="sail-wipe">' +
      '<div class="sail-loader" role="status" aria-live="polite">' +
      '<span class="sail-loader-ring" aria-hidden="true"></span>' +
      '<p class="sail-loader-text">Preparing your boarding pass</p>' +
      '<span class="sail-loader-dots" aria-hidden="true">' +
      "<i></i><i></i><i></i></span>" +
      "</div>" +
      "</div>";

    document.body.appendChild(overlay);
    document.body.classList.add("is-sailing");

    var sceneAnim = load(overlay.querySelector("[data-sail-art]"), lottieSrc);
    if (sceneAnim) sceneAnim.setSpeed(1.8);

    window.requestAnimationFrame(function () {
      overlay.classList.add("is-active");
    });

    var navigated = false;
    function go() {
      if (navigated) return;
      navigated = true;
      try { window.sessionStorage.setItem("titanic:boarded", "1"); } catch (e) {}
      window.location.href = target;
    }

    window.setTimeout(function () {
      overlay.classList.add("is-wiping");
    }, SAIL_WIPE_AT);

    // Leave as soon as the wipe has fully covered the viewport; the prefetch
    // started above usually finishes long before that.
    Promise.all([
      new Promise(function (resolve) {
        window.setTimeout(resolve, SAIL_MIN_NAV);
      }),
      Promise.race([
        warmup,
        new Promise(function (resolve) {
          window.setTimeout(resolve, SAIL_MIN_NAV);
        })
      ])
    ]).then(go);

    // Safety net: never trap the visitor on the transition screen.
    window.setTimeout(go, 7000);
  }

  document.querySelectorAll("[data-set-sail]").forEach(function (trigger) {
    var target =
      trigger.getAttribute("href") ||
      (stage && stage.getAttribute("data-voyage-target")) ||
      "/predict";

    ["pointerenter", "focus", "touchstart"].forEach(function (eventName) {
      trigger.addEventListener(eventName, function () {
        warm(target);
      }, { passive: true });
    });

    trigger.addEventListener("click", function (event) {
      event.preventDefault();
      sailTo(target);
    });
  });

  // Warm the predictor in the background so the first click is instant.
  window.setTimeout(function () {
    var first = document.querySelector("[data-set-sail]");
    var idleTarget =
      (first && first.getAttribute("href")) ||
      (stage && stage.getAttribute("data-voyage-target")) ||
      "/predict";
    warm(idleTarget);
  }, 1200);

  // ------------------------------------------------------- result vignette
  var vignette = document.querySelector("[data-result-voyage]");
  if (vignette) {
    var probability = parseFloat(vignette.getAttribute("data-probability")) || 50;
    var survived = vignette.getAttribute("data-outcome") === "survived";
    var vignetteArt = vignette.querySelector("[data-result-voyage-canvas]");
    var vignetteSrc = vignette.getAttribute("data-voyage-src") || DEFAULT_SRC;
    var vignetteAnim = load(vignetteArt, vignetteSrc);

    if (!vignetteAnim) {
      vignette.hidden = true;
    } else {
      vignette.classList.add("is-live");
      vignette.style.setProperty("--survival", (probability / 100).toFixed(2));
      vignetteAnim.setSpeed(
        reduceMotion ? 0.5 : survived ? 0.55 + (probability / 100) * 1.15 : 0.4
      );

      // Replay the voyage on click, so the animation is explorable.
      vignette.addEventListener("click", function () {
        vignetteAnim.goToAndPlay(0, true);
        vignetteAnim.setSpeed(
          reduceMotion ? 0.5 : survived ? 0.55 + (probability / 100) * 1.15 : 0.4
        );
      });
    }
  }

  // ------------------------------------------- one-time "welcome aboard" toast
  var boarded = false;
  try {
    boarded = window.sessionStorage.getItem("titanic:boarded") === "1";
    if (boarded) window.sessionStorage.removeItem("titanic:boarded");
  } catch (e) {
    boarded = false;
  }

  if (boarded && document.querySelector("[data-predict-page]")) {
    var toast = document.createElement("div");
    toast.className = "toast";
    toast.setAttribute("role", "status");
    toast.innerHTML =
      '<span class="toast-dot" aria-hidden="true"></span>' +
      "<span>Welcome aboard. The 100-tree forest is ready.</span>";
    document.body.appendChild(toast);
    window.requestAnimationFrame(function () {
      toast.classList.add("is-visible");
    });
    window.setTimeout(function () {
      toast.classList.remove("is-visible");
    }, 4200);
  }
})();
