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
  var modelNote = document.getElementById("text-model-note");
  var explBox = document.getElementById("text-explanations");
  var explList = document.getElementById("text-explanations-list");
  var openReport = document.getElementById("text-open-report");
  var lastPayload = null;

  function showErr(msg) {
    errBox.textContent = msg || "An error occurred.";
    errBox.classList.add("is-visible");
  }

  function hideErr() {
    errBox.classList.remove("is-visible");
    errBox.textContent = "";
  }

  function fillExplanations(lines) {
    explList.innerHTML = "";
    (lines || []).forEach(function (line) {
      var li = document.createElement("li");
      li.textContent = line;
      explList.appendChild(li);
    });
    explBox.hidden = !lines || lines.length === 0;
  }

  btnAnalyze.addEventListener("click", function () {
    hideErr();
    resultPanel.classList.remove("is-visible");
    if (openReport) openReport.hidden = true;
    explBox.hidden = true;
    modelNote.hidden = true;
    if (btnSave) {
      btnSave.textContent = "Save to History";
      btnSave.disabled = false;
    }
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
        var label = d.label;
        lastPayload = {
          label: label,
          confidence: d.confidence,
          input_summary: d.input_summary,
          model: d.model,
          processing_time_ms: d.processing_time_ms,
          history_id: d.history_id || null,
        };
        labelEl.textContent = label;
        labelEl.classList.remove("fake", "real", "uncertain");
        if (label === "Fake") labelEl.classList.add("fake");
        else if (label === "Real") labelEl.classList.add("real");
        else labelEl.classList.add("uncertain");
        var pct = Math.round((d.confidence || 0) * 100);
        barFill.style.width = pct + "%";
        barFill.classList.remove("is-fake", "is-real", "is-uncertain");
        if (label === "Fake") barFill.classList.add("is-fake");
        else if (label === "Real") barFill.classList.add("is-real");
        else barFill.classList.add("is-uncertain");
        metaTime.textContent =
          "Processing time: " + (d.processing_time_ms || 0) + " ms";
        if (d.model_label && d.model_label !== label) {
          modelNote.textContent =
            "Raw style vote: " + d.model_label + " (shown as " + label + " after uncertainty rules).";
          modelNote.hidden = false;
        } else {
          modelNote.hidden = true;
        }
        fillExplanations(d.explanations);
        if (openReport) openReport.hidden = false;
        if (d.history_id && btnSave) {
          btnSave.textContent = "Saved to history";
          btnSave.disabled = true;
        } else if (btnSave) {
          btnSave.textContent = "Save to History";
          btnSave.disabled = false;
        }
        resultPanel.classList.add("is-visible");
      })
      .catch(function () {
        loading.classList.remove("is-visible");
        showErr("Could not connect to the server.");
      });
  });

  if (btnSave) {
    btnSave.addEventListener("click", function () {
      if (!lastPayload) return;
      if (lastPayload.history_id) {
        alert("This run is already in Scan History.");
        return;
      }
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
          if (btnSave) {
            lastPayload.history_id = j.data && j.data.id;
            btnSave.textContent = "Saved to history";
            btnSave.disabled = true;
          }
        })
        .catch(function () {
          showErr("Could not save to history.");
        });
    });
  }
})();
