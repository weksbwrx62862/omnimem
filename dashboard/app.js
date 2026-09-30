"use strict";

const CFG = {
  load() {
    return {
      base: localStorage.getItem("omni.base") || "",
      key: localStorage.getItem("omni.key") || "",
      admin: localStorage.getItem("omni.admin") || "",
    };
  },
  save() {
    localStorage.setItem("omni.base", $("cfgBase").value.trim());
    localStorage.setItem("omni.key", $("cfgKey").value.trim());
    localStorage.setItem("omni.admin", $("cfgAdmin").value.trim());
    setStatus("已配置", "ok");
  },
};

function $(id) { return document.getElementById(id); }
function setStatus(text, cls) {
  const el = $("status");
  el.textContent = text;
  el.className = "badge" + (cls ? " " + cls : "");
}

// 同源请求由 omni_dashboard.py 代理到真正的 REST 后端，规避 CORS。
async function api(path, { method = "GET", body, admin = false } = {}) {
  const headers = { Accept: "application/json" };
  const key = (localStorage.getItem("omni.key") || "").trim();
  if (key) headers["Authorization"] = /^Bearer\s/i.test(key) ? key : `Bearer ${key}`;
  const adminTok = (localStorage.getItem("omni.admin") || "").trim();
  if (admin && adminTok) headers["X-Admin-Token"] = adminTok;
  const opts = { method, headers };
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = text; }
  if (!res.ok) throw new Error(`${res.status}: ${data && data.error ? data.error : text}`);
  if (data && typeof data === "object" && data.status === "error") throw new Error(data.reason || "server error");
  return data;
}

function card(inner, extraCls) {
  const div = document.createElement("div");
  div.className = "card" + (extraCls ? " " + extraCls : "");
  div.innerHTML = inner;
  return div;
}

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function renderRecall(data, target) {
  const out = $(target);
  out.innerHTML = "";
  const items = data.results || data.memories || [];
  if (!items.length) { out.appendChild(card("<em>无结果</em>")); return; }
  for (const it of items) {
    const score = typeof it.score === "number" ? it.score.toFixed(3) : "";
    out.appendChild(card(
      `<div class="ct">${esc(it.content || it.summary || "")}</div>` +
      `<div class="meta">${esc(it.type || it.memory_type || "")} · ${esc(it.privacy || "")} ` +
      (score ? `· <b>${score}</b>` : "") +
      (it.id ? ` · <span class="mono">${esc(it.id)}</span>` : "") +
      `</div>` +
      (data._explain ? "" : "")
    ));
  }
}

async function doRecall() {
  setStatus("检索中…");
  try {
    const body = { query: $("qText").value.trim(), mode: $("qMode").value };
    if (!body.query) { setStatus("请输入查询", "err"); return; }
    if ($("qExplain").checked) body.explain = true;
    const data = await api("/api/recall", { method: "POST", body });
    renderRecall(data, "recallOut");
    setStatus(`命中 ${(data.results || data.memories || []).length}`, "ok");
  } catch (e) { setStatus("错误", "err"); alert(String(e.message || e)); }
}

async function doMemorize() {
  const content = $("mContent").value.trim();
  if (!content) { setStatus("内容为空", "err"); return; }
  setStatus("写入中…");
  try {
    const data = await api("/api/memorize", {
      method: "POST",
      body: { content, memory_type: $("mType").value, privacy: $("mPrivacy").value, confidence: Number($("mConf").value) },
    });
    $("memorizeOut").textContent = JSON.stringify(data, null, 2);
    setStatus("已写入", "ok");
  } catch (e) { setStatus("错误", "err"); alert(String(e.message || e)); }
}

async function doDetailList() {
  setStatus("加载中…");
  try {
    const data = await api("/api/detail", { method: "POST", body: { action: "list" } });
    renderRecall(data, "detailOut");
    setStatus("已加载", "ok");
  } catch (e) { setStatus("错误", "err"); alert(String(e.message || e)); }
}

async function doGovern() {
  const action = $("gAction").value;
  let extra = {};
  const raw = $("gArgs").value.trim();
  if (raw) { try { extra = JSON.parse(raw); } catch { alert("附加参数需为合法 JSON"); return; } }
  setStatus("执行中…");
  try {
    const data = await api("/api/govern", { method: "POST", body: { action, ...extra } });
    $("governOut").textContent = JSON.stringify(data, null, 2);
    setStatus("完成", "ok");
  } catch (e) { setStatus("错误", "err"); $("governOut").textContent = String(e.message || e); }
}

async function doSystem(kind) {
  $("systemOut").textContent = "…";
  try {
    if (kind === "metrics") {
      const text = await api("/metrics");
      $("systemOut").textContent = typeof text === "string" ? text : JSON.stringify(text, null, 2);
    } else if (kind === "health") {
      const data = await api("/api/health");
      $("systemOut").textContent = JSON.stringify(data, null, 2);
      setStatus(data.status || "ok", data.status === "healthy" ? "ok" : "err");
    } else {
      const data = await api("/api/tools");
      $("systemOut").textContent = JSON.stringify(data, null, 2);
      populateGovernActions(data.tools || []);
    }
  } catch (e) { $("systemOut").textContent = String(e.message || e); setStatus("错误", "err"); }
}

const GOVERN_ACTIONS = ["health", "audit", "stats", "list", "export_training_data", "register_adapter", "reencrypt", "migrate_index", "consolidate", "forget"];
function populateGovernActions() {
  const sel = $("gAction");
  sel.innerHTML = "";
  for (const a of GOVERN_ACTIONS) sel.appendChild(Object.assign(document.createElement("option"), { value: a, text: a }));
}

function initTabs() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((p) => p.classList.add("hidden"));
      btn.classList.add("active");
      $("tab-" + btn.dataset.tab).classList.remove("hidden");
    });
  });
}

function init() {
  const c = CFG.load();
  $("cfgBase").value = c.base; $("cfgKey").value = c.key; $("cfgAdmin").value = c.admin;
  $("btnSave").addEventListener("click", CFG.save);
  $("btnRecall").addEventListener("click", doRecall);
  $("qText").addEventListener("keydown", (e) => { if (e.key === "Enter") doRecall(); });
  $("btnMemorize").addEventListener("click", doMemorize);
  $("btnDetailList").addEventListener("click", doDetailList);
  $("btnGovern").addEventListener("click", doGovern);
  $("btnHealth").addEventListener("click", () => doSystem("health"));
  $("btnTools").addEventListener("click", () => doSystem("tools"));
  $("btnMetrics").addEventListener("click", () => doSystem("metrics"));
  populateGovernActions();
  initTabs();
}

document.addEventListener("DOMContentLoaded", init);
