(function () {
  var tbody = document.getElementById("history-tbody");
  var filterSel = document.getElementById("history-filter");
  var btnClear = document.getElementById("history-clear-all");
  var emptyMsg = document.getElementById("history-empty");
  var allRows = [];

  function typeLabel(t) {
    if (t === "text") return "Text";
    if (t === "image") return "Image";
    if (t === "video") return "Video";
    return t || "—";
  }

  function render() {
    var f = filterSel.value;
    tbody.innerHTML = "";
    var list = allRows.filter(function (r) {
      if (f === "all") return true;
      return r.detection_type === f;
    });
    if (list.length === 0) {
      emptyMsg.classList.remove("is-hidden");
      return;
    }
    emptyMsg.classList.add("is-hidden");
    list.forEach(function (r) {
      var tr = document.createElement("tr");
      var td1 = document.createElement("td");
      td1.textContent = typeLabel(r.detection_type);
      var td2 = document.createElement("td");
      td2.textContent = r.input_summary || "—";
      var td3 = document.createElement("td");
      var span = document.createElement("span");
      span.textContent = r.result_label;
      span.className =
        r.result_label === "Fake" ? "badge-fake" : "badge-real";
      td3.appendChild(span);
      var td4 = document.createElement("td");
      td4.textContent = (r.confidence_score * 100).toFixed(1) + "%";
      var td5 = document.createElement("td");
      td5.textContent = r.created_at || "—";
      var td6 = document.createElement("td");
      var del = document.createElement("button");
      del.type = "button";
      del.className = "btn-danger-sm";
      del.textContent = "Delete";
      del.dataset.id = r.id;
      del.addEventListener("click", function () {
        if (!confirm("Delete this record?")) return;
        fetch("/history/delete/" + r.id, { method: "DELETE" })
          .then(function (x) {
            return x.json();
          })
          .then(function (j) {
            if (j.success) load();
          });
      });
      td6.appendChild(del);
      tr.appendChild(td1);
      tr.appendChild(td2);
      tr.appendChild(td3);
      tr.appendChild(td4);
      tr.appendChild(td5);
      tr.appendChild(td6);
      tbody.appendChild(tr);
    });
  }

  function load() {
    fetch("/history/data")
      .then(function (r) {
        return r.json();
      })
      .then(function (j) {
        if (!j.success || !j.data || !j.data.records) {
          allRows = [];
        } else {
          allRows = j.data.records;
        }
        render();
      })
      .catch(function () {
        allRows = [];
        render();
      });
  }

  filterSel.addEventListener("change", render);
  btnClear.addEventListener("click", function () {
    if (!confirm("Delete all records?")) return;
    fetch("/history/clear", { method: "DELETE" })
      .then(function (r) {
        return r.json();
      })
      .then(function (j) {
        if (j.success) load();
      });
  });

  load();
})();
