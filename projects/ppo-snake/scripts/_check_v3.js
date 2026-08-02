// 综合校验 v3 讲解页：JS 语法 / 标签配对 / 关键引用 / AI_MODEL 与 TRAIN_DATA
// 用法：node scripts/_check_v3.js
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const file = path.join(__dirname, '..', 'PPO贪吃蛇讲解v3.html');
const html = fs.readFileSync(file, 'utf-8');

let pass = 0, fail = 0;
function ok(cond, msg) {
  if (cond) { pass++; console.log('  ✓ ' + msg); }
  else { fail++; console.log('  ✗ ' + msg); }
}

console.log('== 1. 关键引用 ==');
['var AI_MODEL = {', 'var TRAIN_DATA = {', 'id="gameCanvas"', 'id="trainChart"',
  'id="gaeChart"', 'id="clipChart"', 'id="loopGrid"', 'MathJax'].forEach(k =>
  ok(html.includes(k), `包含 ${k}`));

console.log('== 2. JS 语法（提取所有 <script> 块）==');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
ok(scripts.length >= 3, `找到 ${scripts.length} 个 <script> 块`);
let jsOk = true;
scripts.forEach((s, i) => {
  try { new vm.Script(s); }
  catch (e) { jsOk = false; console.log(`    block ${i} 语法错误: ${e.message}`); }
});
ok(jsOk, '所有 <script> 块语法正确');

console.log('== 3. 标签配对（粗校验）==');
const tags = ['div', 'section', 'table', 'thead', 'tbody', 'tr', 'td', 'th', 'h2', 'h3', 'h4', 'p', 'ul', 'ol', 'li', 'pre', 'span', 'button', 'canvas', 'nav', 'form', 'select', 'option'];
const voids = new Set(['input', 'br', 'img', 'meta', 'link', 'hr']);
let allOk = true;
for (const tag of tags) {
  const open = (html.match(new RegExp('<' + tag + '(\\s|>)', 'g')) || []).length;
  const close = (html.match(new RegExp('</' + tag + '>', 'g')) || []).length;
  if (open !== close) { allOk = false; console.log(`    <${tag}>: 开 ${open} 闭 ${close} 不匹配`); }
}
ok(allOk, '所有非空标签开闭配对');

console.log('== 4. DOM id 唯一性 ==');
const ids = [...html.matchAll(/id="([^"]+)"/g)].map(m => m[1]);
const dup = ids.filter((v, i) => ids.indexOf(v) !== i);
ok(new Set(dup).size === 0, `id 共 ${new Set(ids).size} 个，无重复`);

console.log('== 5. AI_MODEL / TRAIN_DATA 完整性 ==');
const am = html.match(/var AI_MODEL = (\{[\s\S]*?\});/);
if (am) {
  try {
    const M = JSON.parse(am[1]);
    ok(M.h === 64 && M.na === 3 && M.W1.length === 21 && M.W1[0].length === 64, `AI_MODEL 结构正确 (h=${M.h}, na=${M.na}, W1=${M.W1.length}x${M.W1[0].length})`);
  } catch (e) { ok(false, 'AI_MODEL JSON 解析失败: ' + e.message); }
} else ok(false, '未找到 AI_MODEL');
const td = html.match(/var TRAIN_DATA = (\{[\s\S]*?\});/);
if (td) {
  try {
    const D = JSON.parse(td[1]);
    ok(D.iters.length === D.scores.length && D.scores.length === 51, `TRAIN_DATA 51 个评估点 (iters=${D.iters.length}, scores=${D.scores.length})`);
    ok(D.kl.length === 500, `KL 序列 500 个点 (实际 ${D.kl.length})`);
    ok(D.scores[D.scores.length - 1] === 690.402, `最终得分与 CSV 一致 (${D.scores[D.scores.length - 1]})`);
  } catch (e) { ok(false, 'TRAIN_DATA JSON 解析失败: ' + e.message); }
} else ok(false, '未找到 TRAIN_DATA');

console.log(`\n结果: ${pass} 通过, ${fail} 失败`);
process.exit(fail > 0 ? 1 : 0);
