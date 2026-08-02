// 端到端验证「AI 自动玩」修复：mock DOM/canvas 环境下执行完整游戏 IIFE
// 用法：node scripts/_verify_ai_button.js [html文件名]
const fs = require('fs');
const vm = require('vm');

const target = process.argv[2] || 'PPO贪吃蛇讲解v2.html';
const html = fs.readFileSync(require('path').join(__dirname, '..', target), 'utf-8');

// 1) 提取游戏 IIFE（"手动贪吃蛇游戏"注释到"训练曲线"注释之间，从第一个 (function () { 开始）
const start = html.indexOf('手动贪吃蛇游戏');
const end = html.indexOf('训练曲线');
if (start < 0 || end < 0 || end <= start) { console.error('找不到游戏 IIFE 区间'); process.exit(1); }
let gameCode = html.slice(start, end);
const fnStart = gameCode.indexOf('(function () {');
if (fnStart < 0) { console.error('找不到游戏 IIFE 起始'); process.exit(1); }
gameCode = gameCode.slice(fnStart);
const closeAt = gameCode.lastIndexOf('})();');
if (closeAt < 0) { console.error('找不到游戏 IIFE 闭合'); process.exit(1); }
gameCode = gameCode.slice(0, closeAt + 5);

// 2) 提取真实权重
const modelMatch = html.match(/var AI_MODEL = (\{[\s\S]*?\});/);
if (!modelMatch) { console.error('找不到 AI_MODEL'); process.exit(1); }

// 3) mock DOM / canvas
function makeCtx() {
  const counts = {};
  const noop = (name) => (...args) => { counts[name] = (counts[name] || 0) + 1; };
  return {
    fillRect: noop('fillRect'), clearRect: noop('clearRect'), strokeRect: noop('strokeRect'),
    beginPath: noop('beginPath'), moveTo: noop('moveTo'), lineTo: noop('lineTo'),
    arc: noop('arc'), fill: noop('fill'), stroke: noop('stroke'),
    save: noop('save'), restore: noop('restore'), translate: noop('translate'),
    rotate: noop('rotate'), scale: noop('scale'), setLineDash: noop('setLineDash'),
    fillText: noop('fillText'),
    get counts() { return counts; },
  };
}
const elements = {};
function makeEl(id) {
  const el = {
    id, textContent: '', handlers: {},
    width: 460, height: 460,
    addEventListener(type, fn) { this.handlers[type] = fn; },
    getContext() { return el.__ctx || (el.__ctx = makeCtx()); },
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 460, height: 460 }),
  };
  return el;
}
const documentMock = {
  getElementById(id) { return elements[id] || (elements[id] = makeEl(id)); },
};

const sandbox = {
  document: documentMock,
  window: { addEventListener() {} },
  Math, JSON, Set, Array, Number, parseInt, isNaN,
  setInterval, clearInterval, setTimeout, clearTimeout, console,
};
sandbox.window.setInterval = setInterval;
sandbox.window.clearInterval = clearInterval;
vm.createContext(sandbox);

// 先注入 AI_MODEL，再跑游戏代码
vm.runInContext('var AI_MODEL = ' + modelMatch[1] + ';', sandbox);
vm.runInContext(gameCode, sandbox, { filename: target });

// 4) 模拟点击「AI 自动玩」
const gameAI = elements.gameAI;
const gameRestart = elements.gameRestart;
if (!gameAI || !gameAI.handlers.click) { console.error('gameAI 按钮未绑定'); process.exit(1); }

const before = elements.gameCanvas.__ctx.counts.fillRect || 0;
gameAI.handlers.click();
const statusMid = elements.gStatus.textContent;
const countRightAfter = elements.gameCanvas.__ctx.counts.fillRect;

// 跑 ~1.5s（25 帧 @60ms），期间应持续绘制
setTimeout(() => {
  const after = elements.gameCanvas.__ctx.counts.fillRect;
  const status = elements.gStatus.textContent;
const framesDrawn = after - before;
const immediateDraw = countRightAfter > before;
console.log('点击后立即状态:', JSON.stringify(statusMid));
console.log('点击瞬间是否已触发绘制 (fillRect):', immediateDraw ? 'YES' : 'NO');
console.log('1.5 秒后状态:', JSON.stringify(status));
console.log('1.5 秒内新增 fillRect 调用次数:', framesDrawn);
// 一局 AI 要跑几百帧（吃 25+ 个食物），1.5s 内不会结束属正常；
// 关键指标是画布是否在持续绘制（修复前第一帧抛异常，fillRect 为 0）
const ok = framesDrawn >= 10;
console.log('AI 自动玩画面在持续绘制:', ok ? 'YES' : 'NO');
console.log(ok ? '\n=== AI 自动玩修复验证通过 ===' : '\n=== AI 自动玩仍异常（画面未绘制） ===');
process.exit(ok ? 0 : 1);
}, 1500);
