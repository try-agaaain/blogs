// 端到端验证「AI 自动玩」与图表渲染：mock DOM/canvas 环境下执行页面完整脚本
// 支持 v1 / v2 / v3 三种讲解页（自动识别数据块与脚本布局）
// 用法：node scripts/verify_ai_play.js [html文件名]   （默认 PPO贪吃蛇讲解v2.html）
const fs = require('fs');
const vm = require('vm');

const target = process.argv[2] || 'PPO贪吃蛇讲解v2.html';
const html = fs.readFileSync(require('path').join(__dirname, '..', target), 'utf-8');

const isV3 = html.includes('var TRAIN_DATA');

// 1) 提取主逻辑脚本：v3 取"不含 MathJax / AI_MODEL 定义"的内联脚本块；v1/v2 沿用注释区间法
let code;
if (isV3) {
  const blocks = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
  const main = blocks.find(b => !b.includes('MathJax') && !b.includes('var AI_MODEL = '));
  if (!main) { console.error('找不到 v3 主脚本块'); process.exit(1); }
  code = main;
} else {
  const start = html.indexOf('手动贪吃蛇游戏');
  const end = html.indexOf('训练曲线');
  if (start < 0 || end < 0 || end <= start) { console.error('找不到游戏 IIFE 区间'); process.exit(1); }
  let gameCode = html.slice(start, end);
  const fnStart = gameCode.indexOf('(function () {');
  if (fnStart < 0) { console.error('找不到游戏 IIFE 起始'); process.exit(1); }
  gameCode = gameCode.slice(fnStart);
  const closeAt = gameCode.lastIndexOf('})();');
  if (closeAt < 0) { console.error('找不到游戏 IIFE 闭合'); process.exit(1); }
  code = gameCode.slice(0, closeAt + 5);
}

// 2) 提取真实权重与曲线数据
const modelMatch = html.match(/var AI_MODEL = (\{[\s\S]*?\});/);
if (!modelMatch) { console.error('找不到 AI_MODEL'); process.exit(1); }
const dataMatch = html.match(/var TRAIN_DATA = (\{[\s\S]*?\});/);

// 3) mock DOM / canvas
function makeCtx() {
  const counts = {};
  const noop = (name) => (...args) => { counts[name] = (counts[name] || 0) + 1; };
  return {
    fillRect: noop('fillRect'), clearRect: noop('clearRect'), strokeRect: noop('strokeRect'),
    beginPath: noop('beginPath'), closePath: noop('closePath'), moveTo: noop('moveTo'), lineTo: noop('lineTo'),
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
    id, textContent: '', handlers: {}, value: '0.95',
    width: 460, height: 460, clientWidth: 460, clientHeight: 460,
    addEventListener(type, fn) { this.handlers[type] = fn; },
    getContext() { return el.__ctx || (el.__ctx = makeCtx()); },
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 460, height: 460 }),
    getAttribute(name) { return name === 'height' ? String(this.height) : null; },
    appendChild() {},
    classList: { add() {}, remove() {}, toggle() {} },
  };
  return el;
}
const documentMock = {
  getElementById(id) { return elements[id] || (elements[id] = makeEl(id)); },
  querySelectorAll() { return []; },
  createElement() { return makeEl('dyn-' + (documentMock.__n = (documentMock.__n || 0) + 1)); },
};

const sandbox = {
  document: documentMock,
  window: { addEventListener() {}, devicePixelRatio: 1, innerWidth: 1200, innerHeight: 800 },
  Math, JSON, Set, Array, Number, parseInt, isNaN,
  setInterval, clearInterval, setTimeout, clearTimeout, console,
  IntersectionObserver: class { observe() {} },
  requestAnimationFrame: () => 0,
};
sandbox.window.setInterval = setInterval;
sandbox.window.clearInterval = clearInterval;
vm.createContext(sandbox);

// 注入数据块，再执行主脚本
vm.runInContext('var AI_MODEL = ' + modelMatch[1] + ';', sandbox);
if (dataMatch) vm.runInContext('var TRAIN_DATA = ' + dataMatch[1] + ';', sandbox);
vm.runInContext(code, sandbox, { filename: target });

// 4) 验证：AI 按钮触发实时推理并持续绘制
const gameAI = elements.gameAI;
const gameRestart = elements.gameRestart;
if (!gameAI || !gameAI.handlers.click) { console.error('gameAI 按钮未绑定'); process.exit(1); }

const before = elements.gameCanvas.__ctx.counts.fillRect || 0;
gameAI.handlers.click();
const statusMid = elements.gStatus.textContent;
const countRightAfter = elements.gameCanvas.__ctx.counts.fillRect;

// 图表是否绘制（有 canvas 上下文调用即视为执行）
const gaeDrawn = (elements.gaeChart && elements.gaeChart.__ctx && elements.gaeChart.__ctx.counts.fillRect > 0);
const clipDrawn = (elements.clipChart && elements.clipChart.__ctx && elements.clipChart.__ctx.counts.fillRect > 0);
const trainDrawn = (elements.trainChart && elements.trainChart.__ctx && (elements.trainChart.__ctx.counts.fillRect > 0 || elements.trainChart.__ctx.counts.stroke > 0));

setTimeout(() => {
  const after = elements.gameCanvas.__ctx.counts.fillRect;
  const status = elements.gStatus.textContent;
  const framesDrawn = after - before;
  const immediateDraw = countRightAfter > before;
  const gaeOk = !isV3 || gaeDrawn;
  const clipOk = !isV3 || clipDrawn;
  const trainOk = !isV3 || trainDrawn;
  console.log('点击后立即状态:', JSON.stringify(statusMid));
  console.log('点击瞬间是否已触发绘制 (fillRect):', immediateDraw ? 'YES' : 'NO');
  console.log('1.5 秒后状态:', JSON.stringify(status));
  console.log('1.5 秒内新增 fillRect 调用次数:', framesDrawn);
  if (isV3) {
    console.log('GAE 图已绘制:', gaeDrawn ? 'YES' : 'NO');
    console.log('Clip 图已绘制:', clipDrawn ? 'YES' : 'NO');
    console.log('训练曲线已绘制:', trainDrawn ? 'YES' : 'NO');
  }
  const ok = framesDrawn >= 10 && gaeOk && clipOk && trainOk;
  console.log(ok ? '\n=== AI 自动玩 + 图表渲染验证通过 ===' : '\n=== 仍有异常 ===');
  process.exit(ok ? 0 : 1);
}, 1500);
