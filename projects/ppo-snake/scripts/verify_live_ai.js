// 端到端验证：模拟 HTML 中实时 AI 决策的完整逻辑（不依赖 DOM）
// 从 HTML 中提取 AI_MODEL 与 aiGetState / aiForward / aiAct 函数来跑
// 支持 v1 / v2 / v3 三种讲解页（自动检测初始蛇位与 N 依赖）
// 用法：node scripts/verify_live_ai.js [html文件名]   （默认 PPO贪吃蛇讲解.html）
const fs = require('fs');
const vm = require('vm');

const target = process.argv[2] || 'PPO贪吃蛇讲解.html';
const html = fs.readFileSync(require('path').join(__dirname, '..', target), 'utf-8');

const isV3 = html.includes('var TRAIN_DATA');

// 提取 AI_MODEL 定义（内嵌为 JSON 对象，可能包含嵌套数组）
const modelMatch = html.match(/var AI_MODEL = (\{[\s\S]*?\});/);
if (!modelMatch) { console.error('找不到 AI_MODEL'); process.exit(1); }
const aiModelJson = modelMatch[1];

// 提取 aiGetState / aiForward / aiAct 函数定义
function extractFn(name) {
  const re = new RegExp('function ' + name + '\\([\\s\\S]*?\\n  \\}');
  const m = html.match(re);
  if (!m) { console.error('找不到函数', name); process.exit(1); }
  return m[0];
}

const fns = ['aiGetState', 'aiForward', 'aiAct'];
const fnSrc = fns.map(extractFn).join('\n');
// 函数依赖注入：v3 的 aiGetState 引用外部 N（网格边长）；v1/v2 引用外部 DIRS
const needsN = !/const N = |let N = /.test(fnSrc) && fnSrc.includes('N');
const DIRS_SRC = fnSrc.includes('DIRS') && !fnSrc.includes('const DIRS') && !fnSrc.includes('DIR_VEC')
  ? 'const DIRS = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };' : '';
const EXTRA = (needsN ? 'const N = 12;' : '') + (DIRS_SRC ? '\n' + DIRS_SRC : '');

const sandbox = { Math, console, Set, Array, Number, parseInt, isNaN };
sandbox.AI_MODEL = undefined;
vm.createContext(sandbox);
vm.runInContext('var AI_MODEL = ' + aiModelJson + ';\n' + EXTRA + '\n' + fnSrc, sandbox);

const M = sandbox.AI_MODEL;
console.log(`[${target}] 模型维度: h=${M.h} na=${M.na} W1=${M.W1.length}x${M.W1[0].length}`);

// 蛇游戏模拟（初始蛇位与对应页面一致：v3 对齐 snake_game.py 的 (4,6)）
const N = 12;
const DIR_VEC = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
const TURN_A = { up: ['up', 'left', 'right'], down: ['down', 'right', 'left'], left: ['left', 'down', 'up'], right: ['right', 'up', 'down'] };
const START = isV3 ? [[4, 6], [3, 6], [2, 6]] : [[5, 6], [4, 6], [3, 6]];
function playOnce() {
  let snake = START.map(p => p.slice()), dir = 'right';
  let foods = 0;
  function spawnFood(snakeArr) {
    const occ = new Set(snakeArr.map(p => p[0] + ',' + p[1]));
    const free = [];
    for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) if (!occ.has(x + ',' + y)) free.push([x, y]);
    return free[Math.floor(Math.random() * free.length)];
  }
  let food = spawnFood(snake);
  let steps = 0;
  while (steps++ < 5000) {
    const a = sandbox.aiAct(snake, dir, food, M);
    dir = TURN_A[dir][a];
    const [dx, dy] = DIR_VEC[dir];
    const h = snake[0];
    const nx = h[0] + dx, ny = h[1] + dy;
    const hitWall = nx < 0 || nx >= N || ny < 0 || ny >= N;
    const hitBody = snake.some(p => p[0] === nx && p[1] === ny);
    if (hitWall || hitBody) break;
    snake.unshift([nx, ny]);
    if (nx === food[0] && ny === food[1]) { foods++; food = spawnFood(snake); }
    else snake.pop();
  }
  return { foods, len: snake.length };
}

let totalFoods = 0, totalLen = 0, lens = [];
for (let ep = 0; ep < 10; ep++) {
  const r = playOnce();
  totalFoods += r.foods; totalLen += r.len; lens.push(r.len);
  console.log(`局${ep + 1}: 食物=${r.foods} 身长=${r.len}`);
}
console.log(`\n10局均值: 食物=${(totalFoods / 10).toFixed(1)} 身长=${(totalLen / 10).toFixed(1)}`);
console.log(`身长集合: ${lens.join(', ')}`);
console.log('身长是否多样化:', new Set(lens).size > 1 ? '✓ 是' : '✗ 每次都一样');
