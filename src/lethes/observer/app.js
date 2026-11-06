/* The observer is served by lethes itself. No CDN, remote scripts or stored key. */
const byId = (id) => document.getElementById(id);
const state = { key: "", items: [], selected: null, pending: false };
const number = (value) =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString()
    : "—";
const sources = {
  local_estimate: "估算",
  reported_usage: "历史实测",
  reported_prefix_plus_estimate: "实测+估算",
  usage_calibrated_estimate: "校准估算",
};
const reasons = {
  initial: "初建",
  append: "续接",
  replay: "复用",
  capacity: "容量重建",
  recall_capacity: "召回重建",
  economic: "经济重建",
  count_capacity: "计数修复",
};

async function api(path) {
  const response = await fetch(path, {
    headers: { Authorization: `Bearer ${state.key}` },
    cache: "no-store",
  });
  if (!response.ok)
    throw new Error(
      response.status === 401
        ? "API Key 不正确"
        : `服务返回 ${response.status}`,
    );
  return response.json();
}
function cell(row, value, sub) {
  const element = document.createElement("td");
  element.textContent = value;
  if (sub) {
    const small = document.createElement("small");
    small.textContent = sub;
    element.append(small);
  }
  row.append(element);
  return element;
}
function render() {
  const query = byId("search").value.trim().toLowerCase();
  const rows = byId("rows");
  rows.replaceChildren();
  const filtered = state.items.filter(
    (item) =>
      !query ||
      [
        item.observation.model,
        item.observation.request_id,
        item.observation.epoch,
        item.observation.reason,
        item.scope,
      ].some((v) =>
        String(v || "")
          .toLowerCase()
          .includes(query),
      ),
  );
  for (const item of filtered) {
    const o = item.observation,
      usage = item.usage;
    const row = document.createElement("tr");
    row.tabIndex = 0;
    row.onclick = () => showDetail(o.request_id);
    row.onkeydown = (event) => {
      if (event.key === "Enter") showDetail(o.request_id);
    };
    cell(row, new Date(item.created * 1000).toLocaleTimeString());
    cell(
      row,
      o.model || "—",
      `${o.request_id.slice(0, 12)} · ${item.scope.slice(0, 8)}`,
    );
    const budget = cell(
      row,
      `${number(o.input_tokens)} / ${number(o.input_limit)}`,
    );
    const bar = document.createElement("span");
    bar.className = "bar";
    const fill = document.createElement("span");
    fill.style.width = `${Math.max(0, Math.min(100, (100 * o.input_tokens) / o.input_limit))}%`;
    bar.append(fill);
    budget.append(bar);
    cell(row, sources[o.count_source] || o.count_source);
    cell(
      row,
      item.authoritative
        ? number(usage.input_tokens)
        : usage?.aggregate
          ? "整轮汇总"
          : "—",
    );
    cell(row, item.authoritative ? number(usage.cached_tokens) : "—");
    cell(
      row,
      `${number(o.recalled_messages)} / ${number(o.omitted_messages)}`,
    ).className = o.omitted_messages ? "warning" : "";
    cell(row, reasons[o.reason] || o.reason, o.epoch.slice(0, 10));
    cell(row, `${number(o.elapsed_ms)} ms`);
    rows.append(row);
  }
  if (!rows.children.length) {
    const row = document.createElement("tr");
    const td = cell(
      row,
      "暂无匹配请求；在 Open WebUI 启用远程 Filter 后开始聊天。",
    );
    td.colSpan = 9;
    rows.append(row);
  }
  const measured = filtered.filter((i) => i.authoritative);
  byId("requests").textContent = number(filtered.length);
  byId("measured").textContent = number(measured.length);
  const caches = measured
    .map((i) => i.usage.cached_tokens)
    .filter((v) => typeof v === "number");
  byId("cache").textContent = caches.length
    ? number(caches.reduce((a, b) => a + b, 0))
    : "—";
  const times = filtered
    .map((i) => i.observation.elapsed_ms)
    .filter((v) => typeof v === "number")
    .sort((a, b) => a - b);
  const mid = Math.floor(times.length / 2);
  byId("latency").textContent = times.length
    ? `${number(times.length % 2 ? times[mid] : (times[mid - 1] + times[mid]) / 2)} ms`
    : "—";
}
async function showDetail(id) {
  state.selected = id;
  try {
    const detail = await api(`/v2/observations/${encodeURIComponent(id)}`);
    if (state.selected !== id) return;
    byId("detail-json").textContent = JSON.stringify(detail, null, 2);
    byId("detail").hidden = false;
  } catch (error) {
    byId("connection").textContent = error.message;
  }
}
async function refresh() {
  if (!state.key || state.pending) return;
  state.pending = true;
  try {
    const data = await api("/v2/observations?limit=100");
    state.items = data.items;
    render();
    byId("connection").textContent =
      `已连接 · ${new Date().toLocaleTimeString()}`;
    byId("connection").className = "good";
    if (state.selected) await showDetail(state.selected);
  } catch (error) {
    byId("connection").textContent = error.message;
    byId("connection").className = "warning";
  } finally {
    state.pending = false;
  }
}
byId("connect").onsubmit = (event) => {
  event.preventDefault();
  state.key = byId("key").value;
  byId("key").value = "";
  refresh();
};
byId("refresh").onclick = refresh;
byId("search").oninput = render;
byId("close").onclick = () => {
  state.selected = null;
  byId("detail").hidden = true;
};
setInterval(refresh, 2000);
