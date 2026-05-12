(function () {
  var drop = document.getElementById("image-drop");
  var input = document.getElementById("image-input");
  var btnAnalyze = document.getElementById("btn-analyze-image");
  var btnSave = document.getElementById("btn-save-image");
  var preview = document.getElementById("image-preview");
  var previewImg = document.getElementById("image-preview-img");
  var meta = document.getElementById("image-file-meta");
  var loading = document.getElementById("image-loading");
  var resultPanel = document.getElementById("image-result");
  var errBox = document.getElementById("image-error");
  var labelEl = document.getElementById("image-result-label");
  var barFill = document.getElementById("image-confidence-fill");
  var metaTime = document.getElementById("image-meta-time");
  var modelNote = document.getElementById("image-model-note");
  var elaWrap = document.getElementById("image-ela-wrap");
  var elaImg = document.getElementById("image-ela-img");
  var explBox = document.getElementById("image-explanations");
  var explList = document.getElementById("image-explanations-list");
  var openReport = document.getElementById("image-open-report");
  var fileObj = null;
  var lastPayload = null;
  var maxBytes = 20 * 1024 * 1024;

  function showErr(msg) {
    errBox.textContent = msg || "An error occurred.";
    errBox.classList.add("is-visible");
  }

  function hideErr() {
    errBox.classList.remove("is-visible");
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

  drop.addEventListener("click", function () {
    input.click();
  });
  input.addEventListener("change", function () {
    if (input.files && input.files[0]) setFile(input.files[0]);
  });
  ["dragenter", "dragover"].forEach(function (ev) {
    drop.addEventListener(ev, function (e) {
      e.preventDefault();
      drop.classList.add("is-dragover");
    });
  });
  ["dragleave", "drop"].forEach(function (ev) {
    drop.addEventListener(ev, function (e) {
      e.preventDefault();
      drop.classList.remove("is-dragover");
    });
  });
  drop.addEventListener("drop", function (e) {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  });

  function setFile(f) {
    if (!f) return;
    if (f.size > maxBytes) {
      showErr("Image size exceeds 20 MB.");
      return;
    }
    hideErr();
    fileObj = f;
    var url = URL.createObjectURL(f);
    previewImg.src = url;
    preview.classList.add("is-visible");
    meta.textContent =
      f.name + " — " + (f.size / 1024 / 1024).toFixed(2) + " MB";
  }

  btnAnalyze.addEventListener("click", function () {
    if (!fileObj) {
      showErr("Please select an image first.");
      return;
    }
    hideErr();
    resultPanel.classList.remove("is-visible");
    elaWrap.hidden = true;
    explBox.hidden = true;
    if (modelNote) modelNote.hidden = true;
    if (openReport) openReport.hidden = true;
    if (btnSave) {
      btnSave.textContent = "Save to History";
      btnSave.disabled = false;
    }
    loading.classList.add("is-visible");
    var fd = new FormData();
    fd.append("file", fileObj);
    fetch("/detect/image/analyze", { method: "POST", body: fd })
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
          file_name: d.file_name,
          model: d.model,
          processing_time_ms: d.processing_time_ms,
          history_id: d.history_id || null,
        };
        var label = d.label;
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
        if (d.model_label && d.model_label !== label && modelNote) {
          modelNote.textContent =
            "Model vote: " + d.model_label + " (shown as " + label + ").";
          modelNote.hidden = false;
        } else if (modelNote) {
          modelNote.hidden = true;
        }
        fillExplanations(d.explanations);
        if (d.ela_image && elaImg && elaWrap) {
          elaImg.src = d.ela_image;
          elaWrap.hidden = false;
        } else if (elaWrap) {
          elaWrap.hidden = true;
        }
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
      fetch("/detect/image/save", {
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
