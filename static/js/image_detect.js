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

  btnAnalyze.addEventListener("click", function () {
    if (!fileObj) {
      showErr("Please select an image first.");
      return;
    }
    hideErr();
    resultPanel.classList.remove("is-visible");
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
      })
      .catch(function () {
        showErr("Could not save to history.");
      });
  });
})();
