// 端到端验证：模拟 HTML 中实时 AI 决策的完整逻辑（不依赖 DOM）
// 从 HTML 中提取 playAI 相关的辅助函数（aiGetState / aiForward / aiAct）来跑
const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync(require('path').join(__dirname, '..', 'PPO贪吃蛇讲解.html'), 'utf-8');

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
// aiGetState 依赖 DIRS（方向增量表），一并注入
const DIRS_SRC = 'const DIRS = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };';

const sandbox = { Math, console, Set, Array, Number, parseInt, isNaN };
sandbox.AI_MODEL = undefined;
vm.createContext(sandbox);
vm.runInContext('var AI_MODEL = ' + aiModelJson + ';\n' + DIRS_SRC + '\n' + fnSrc, sandbox);

const M = sandbox.AI_MODEL;
console.log('模型维度: h=' + M.h, 'na=' + M.na, 'W1 行数=' + M.W1.length, '列数=' + M.W1[0].length);

// 蛇游戏模拟（与 HTML step 一致）
const N = 12;
const DIR_VEC = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
const TURN_A = { up: ['up', 'left', 'right'], down: ['down', 'right', 'left'], left: ['left', 'down', 'up'], right: ['right', 'up', 'down'] };
function playOnce() {
  let snake = [[5, 6], [4, 6], [3, 6]], dir = 'right';
  let foods = 0;
  const occ = new Set(snake.map(p => p[0] + ',' + p[1]));
  const free = [];
  for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) if (!occ.has(x + ',' + y)) free.push([x, y]);
  let food = free[Math.floor(Math.random() * free.length)];
  let steps = 0;
  while (steps++ < 3000) {
    const a = sandbox.aiAct(snake, dir, food, M);
    dir = TURN_A[dir][a];
    const [dx, dy] = DIR_VEC[dir];
    const h = snake[0];
    const nx = h[0] + dx, ny = h[1] + dy;
    const hitWall = nx < 0 || nx >= N || ny < 0 || ny >= N;
    const hitBody = snake.some(p => p[0] === nx && p[1] === ny);
    if (hitWall || hitBody) break;
    snake.unshift([nx, ny]);
    if (nx === food[0] && ny === food[1]) {
      foods++;
      const o = new Set(snake.map(p => p[0] + ',' + p[1]));
      const f2 = [];
      for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) if (!o.has(x + ',' + y)) f2.push([x, y]);
      food = f2[Math.floor(Math.random() * f2.length)];
    } else snake.pop();
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
