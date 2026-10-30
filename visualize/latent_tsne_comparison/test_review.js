// Exercise the actual offline review script with a small DOM adapter, without changing real labels.
// Usage: node test_review.js [generated_review.html] [temporary_export.csv]
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { Blob } = require("node:buffer");

async function main() {
  const page = process.argv[2] || path.join(__dirname, "artifacts/draft/review.html");
  const html = fs.readFileSync(page, "utf8");
  const payload = html.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1];
  const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
  const elements = new Map(), stored = new Map();
  const finite = (...values) => values.forEach(v => assert.ok(Number.isFinite(v), "Invalid canvas coordinate"));
  const context2d = Object.fromEntries(["scale", "clearRect", "moveTo", "lineTo", "arc"].map(k => [k, finite]));
  Object.assign(context2d, {beginPath() {}, stroke() {}, fill() {}});
  function element(id) {
    if (!elements.has(id)) elements.set(id, {value: "", checked: false, textContent: "", style: {},
      clientWidth: 600, clientHeight: 420, appendChild() {}, replaceChildren() {}, getContext: () => context2d,
      click() { if (this.onclick) this.onclick(); }});
    return elements.get(id);
  }
  element("data").textContent = payload;
  element("yaw").value = "35"; element("speed").value = "1"; element("follow").checked = true;
  let exported;
  const sandbox = {console, Blob, window: {devicePixelRatio: 1},
    document: {getElementById: element, createElement: () => element("dynamic-" + Math.random()), addEventListener() {}},
    localStorage: {getItem: k => stored.get(k) || null, setItem: (k, v) => stored.set(k, v)},
    URL: {createObjectURL: blob => { exported = blob; return "blob:test"; }, revokeObjectURL() {}},
    setTimeout: callback => callback(), requestAnimationFrame() {}};
  vm.createContext(sandbox);
  vm.runInContext(script, sandbox);
  const count = vm.runInContext("data.samples.length", sandbox);
  assert.ok(count >= 12);
  vm.runInContext("draw(100)", sandbox);
  element("accept").click();
  assert.equal(vm.runInContext("current().s.review_status", sandbox), "accepted");
  element("label").onchange({target: {value: "run"}});
  assert.equal(vm.runInContext("current().s.review_status", sandbox), "pending", "Editing a label must invalidate acceptance");
  vm.runInContext("show(0); persist({label:data.samples[0].row.label,review_status:'pending'})", sandbox);
  vm.runInContext(`for (const c of data.classes) {
    const selected = data.samples.map((s,i)=>s.row.label===c.name?i:-1).filter(i=>i>=0).slice(0,2);
    for (const i of selected) { show(i); persist({review_status:'accepted'}); }
  }`, sandbox);
  element("export").click();
  assert.ok(exported instanceof Blob);
  const csv = await exported.text();
  assert.ok(csv.includes('"review_status"'));
  const accepted = vm.runInContext("data.samples.filter(item=>(saved[item.row.sample_id]||item.row).review_status==='accepted').length", sandbox);
  assert.equal(accepted, 12);
  if (process.argv[3]) fs.writeFileSync(process.argv[3], csv, "utf8");
  console.log(`Review runtime, finite canvas projection, acceptance reset and CSV export passed (${count} motions; 12 simulated decisions in memory only).`);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
