(function () {
  var textarea = document.getElementById("news-text");
  var btnAnalyze = document.getElementById("btn-analyze-text");
  var btnSave = document.getElementById("btn-save-text");
  var loading = document.getElementById("text-loading");
  var resultPanel = document.getElementById("text-result");
  var errBox = document.getElementById("text-error");
  var labelEl = document.getElementById("text-result-label");
  var barFill = document.getElementById("text-confidence-fill");
  var metaTime = document.getElementById("text-meta-time");
  var lastPayload = null;

  function showErr(msg) {
    errBox.textContent = msg || "An error occurred.";
    errBox.classList.add("is-visible");
  }

  function hideErr() {
    errBox.classList.remove("is-visible");
    errBox.textContent = "";
  }

  btnAnalyze.addEventListener("click", function () {
    hideErr();
    resultPanel.classList.remove("is-visible");
    loading.classList.add("is-visible");
    var text = (textarea && textarea.value) || "";
    fetch("/detect/text/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
    })
      .then(function (r) {
        return r.json();
      })
      .then(function (j) {
        loading.classList.remove("is-visible");
        if (!j.success) {
          showErr(j.error || "Analysis failed.");
          return;
        }
        var d = j.data;
        lastPayload = {
          label: d.label,
          confidence: d.confidence,
          input_summary: d.input_summary,
          model: d.model,
          processing_time_ms: d.processing_time_ms,
        };
        labelEl.textContent = d.label;
        labelEl.classList.remove("fake", "real");
        labelEl.classList.add(d.label === "Fake" ? "fake" : "real");
        var pct = Math.round((d.confidence || 0) * 100);
        barFill.style.width = pct + "%";
        barFill.classList.remove("is-fake", "is-real");
        barFill.classList.add(d.label === "Fake" ? "is-fake" : "is-real");
        metaTime.textContent =
          "Processing time: " + (d.processing_time_ms || 0) + " ms";
        resultPanel.classList.add("is-visible");
      })
      .catch(function () {
        loading.classList.remove("is-visible");
        showErr("Could not connect to the server.");
      });
  });

  btnSave.addEventListener("click", function () {
    if (!lastPayload) return;
    fetch("/detect/text/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(lastPayload),
    })
      .then(function (r) {
        return r.json();
      })
      .then(function (j) {
        if (!j.success) {
          showErr(j.error || "Could not save.");
          return;
        }
        alert("Result saved to history.");
      })
      .catch(function () {
        showErr("Could not save to history.");
      });
  });
})();
