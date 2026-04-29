(function () {
  var drop = document.getElementById("video-drop");
  var input = document.getElementById("video-input");
  var btnAnalyze = document.getElementById("btn-analyze-video");
  var btnSave = document.getElementById("btn-save-video");
  var preview = document.getElementById("video-preview");
  var previewVid = document.getElementById("video-preview-el");
  var meta = document.getElementById("video-file-meta");
  var loading = document.getElementById("video-loading");
  var progWrap = document.getElementById("video-progress-wrap");
  var progFill = document.getElementById("video-progress-fill");
  var resultPanel = document.getElementById("video-result");
  var errBox = document.getElementById("video-error");
  var labelEl = document.getElementById("video-result-label");
  var barFill = document.getElementById("video-confidence-fill");
  var metaTime = document.getElementById("video-meta-time");
  var framesLine = document.getElementById("video-frames-line");
  var fileObj = null;
  var lastPayload = null;
  var maxBytes = 100 * 1024 * 1024;
  var progTimer = null;

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
      showErr("Video size exceeds 100 MB.");
      return;
    }
    hideErr();
    fileObj = f;
    var url = URL.createObjectURL(f);
    previewVid.src = url;
    preview.classList.add("is-visible");
    meta.textContent =
      f.name +
      " — " +
      (f.size / 1024 / 1024).toFixed(2) +
      " MB";
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
      showErr("Please select a video file first.");
      return;
    }
    hideErr();
    resultPanel.classList.remove("is-visible");
    loading.classList.add("is-visible");
    progWrap.classList.add("is-visible");
    var w = 12;
    progFill.style.width = w + "%";
    if (progTimer) clearInterval(progTimer);
    progTimer = setInterval(function () {
      w = Math.min(92, w + Math.random() * 8);
      progFill.style.width = w + "%";
    }, 400);

    var fd = new FormData();
    fd.append("file", fileObj);
    fetch("/detect/video/analyze", { method: "POST", body: fd })
      .then(function (r) {
        return r.text().then(function (txt) {
          try {
            return JSON.parse(txt);
          } catch (parseErr) {
            var err = new Error("bad-json");
            err.status = r.status;
            err.body = txt;
            throw err;
          }
        });
      })
      .then(function (j) {
        if (progTimer) {
          clearInterval(progTimer);
          progTimer = null;
        }
        progFill.style.width = "100%";
        setTimeout(function () {
          progWrap.classList.remove("is-visible");
          progFill.style.width = "12%";
        }, 500);
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
        framesLine.textContent =
          "Frames analyzed: " + (d.frames_analyzed || 0);
        resultPanel.classList.add("is-visible");
      })
      .catch(function (err) {
        if (progTimer) {
          clearInterval(progTimer);
          progTimer = null;
        }
        progWrap.classList.remove("is-visible");
        loading.classList.remove("is-visible");
        if (err && err.message === "bad-json") {
          showErr(
            "Unexpected response from the server (the server may have restarted or an internal error occurred). Reload the page and try again."
          );
          return;
        }
        showErr(
          "Could not complete the request. Check that the application is running and that disk space is available, then try again."
        );
      });
  });

  btnSave.addEventListener("click", function () {
    if (!lastPayload) return;
    fetch("/detect/video/save", {
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
