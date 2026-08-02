// 冒烟测试：用 HTML 中内嵌的 AI_MODEL 权重模拟浏览器 AI 玩（采样 + 哈密顿安全滤波兜底）
import fs from 'fs';

const file = process.argv[2] || 'PPO贪吃蛇讲解v3.html';
const html = fs.readFileSync(file, 'utf8');
const mModel = html.match(/var AI_MODEL = (\{[\s\S]*?\});/);
if (!mModel) { console.error('未找到 AI_MODEL'); process.exit(1); }
const M = JSON.parse(mModel[1]);
console.log('h =', M.h, 'na =', M.na, 'W1 rows =', M.W1.length, 'cols =', M.W1[0].length);

const N = 12;
const HAM_DIRS = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
const HAM_TURN = {
  up: { 0: 'up', 1: 'left', 2: 'right' },
  down: { 0: 'down', 1: 'right', 2: 'left' },
  left: { 0: 'left', 1: 'down', 2: 'up' },
  right: { 0: 'right', 1: 'up', 2: 'down' },
};
const HAM_ACTION_TO = {};
for (const d in HAM_TURN) { HAM_ACTION_TO[d] = {}; for (const a in HAM_TURN[d]) HAM_ACTION_TO[d][HAM_TURN[d][a]] = +a; }
function hamBuildCycle(n) {
  const order = [];
  for (let y = 0; y < n; y++) {
    if (y % 2 === 0) { for (let x = 1; x < n; x++) order.push([x, y]); }
    else { for (let x = n - 1; x >= 1; x--) order.push([x, y]); }
  }
  for (let y = n - 1; y >= 0; y--) order.push([0, y]);
  return order;
}
function hamCycleDirs(path) {
  const dirs = [];
  for (let i = 0; i < path.length; i++) {
    const a = path[i], b = path[(i + 1) % path.length];
    const dx = b[0] - a[0], dy = b[1] - a[1];
    dirs.push(dx === 1 ? 'right' : dx === -1 ? 'left' : dy === 1 ? 'down' : 'up');
  }
  return dirs;
}
function hamActionTowards(dir, head, target) {
  const dx = target[0] - head[0], dy = target[1] - head[1];
  const absDir = dx === 1 ? 'right' : dx === -1 ? 'left' : dy === 1 ? 'down' : 'up';
  const a = HAM_ACTION_TO[dir][absDir];
  return a === undefined ? 0 : a;
}
function hamCycleAction(snake, dir, path, pos) {
  const head = snake[0];
  const k = pos.get(head[0] + ',' + head[1]);
  const nxt = path[(k + 1) % path.length];
  return hamActionTowards(dir, head, nxt);
}
function hamSimulate(snake, dir, a, food, N) {
  const nd = HAM_TURN[dir][a];
  const [dx, dy] = HAM_DIRS[nd];
  const [hx, hy] = snake[0];
  const nx = hx + dx, ny = hy + dy;
  if (nx < 0 || nx >= N || ny < 0 || ny >= N) return { ns: snake, nd: dir, ate: false, dead: true };
  if (snake.some(p => p[0] === nx && p[1] === ny)) return { ns: snake, nd: dir, ate: false, dead: true };
  const ate = (nx === food[0] && ny === food[1]);
  const ns = [[nx, ny]].concat(snake);
  if (!ate) ns.pop();
  return { ns, nd, ate, dead: false };
}
function hamBodyOnCycle(snake, path, pos) {
  const n = path.length;
  const k = pos.get(snake[0][0] + ',' + snake[0][1]);
  for (let i = 0; i < snake.length; i++) {
    const idx = pos.get(snake[i][0] + ',' + snake[i][1]);
    if (idx !== (((k - i) % n) + n) % n) return false;
  }
  return true;
}
function hamWalkCycle(snake, dir, path, pos, pathDir, steps, N) {
  const n = path.length;
  let s = snake.slice();
  const body = new Set(s.map(p => p[0] + ',' + p[1]));
  let d = dir;
  for (let t = 0; t < steps; t++) {
    const k = pos.get(s[0][0] + ',' + s[0][1]);
    const nxt = path[(k + 1) % n];
    const nd = pathDir[k];
    const a = HAM_ACTION_TO[d][nd];
    if (a === undefined) return false;
    const nx = nxt[0], ny = nxt[1];
    if (nx < 0 || nx >= N || ny < 0 || ny >= N || body.has(nx + ',' + ny)) return false;
    body.delete(s[s.length - 1][0] + ',' + s[s.length - 1][1]);
    s.pop();
    s.unshift([nx, ny]);
    body.add(nx + ',' + ny);
    d = nd;
  }
  return true;
}
function hamCanEatViaCycle(snake, dir, food, path, pos, pathDir, N) {
  const n = path.length;
  let s = snake.slice();
  const body = new Set(s.map(p => p[0] + ',' + p[1]));
  let d = dir;
  for (let t = 0; t < n + 30; t++) {
    const k = pos.get(s[0][0] + ',' + s[0][1]);
    const nd = pathDir[k];
    const a = HAM_ACTION_TO[d][nd];
    if (a === undefined) return false;
    const nx = path[(k + 1) % n][0], ny = path[(k + 1) % n][1];
    if (body.has(nx + ',' + ny)) return false;
    if (nx === food[0] && ny === food[1]) return true;
    body.delete(s[s.length - 1][0] + ',' + s[s.length - 1][1]);
    s.pop();
    s.unshift([nx, ny]);
    body.add(nx + ',' + ny);
    d = nd;
  }
  return true;
}
function hamSafeMoves(snake, dir, food, path, pos, pathDir, N) {
  if (!food) return new Set([hamCycleAction(snake, dir, path, pos)]);
  const safe = new Set();
  const aligned = hamBodyOnCycle(snake, path, pos);
  const cyc = hamCycleAction(snake, dir, path, pos);
  for (let a = 0; a < 3; a++) {
    const r = hamSimulate(snake, dir, a, food, N);
    if (r.dead) continue;
    if (a === cyc) {
      if (aligned || hamWalkCycle(r.ns, r.nd, path, pos, pathDir, snake.length + 2, N)) safe.add(a);
      continue;
    }
    if (aligned && hamCanEatViaCycle(r.ns, r.nd, food, path, pos, pathDir, N)) safe.add(a);
  }
  if (safe.size === 0) safe.add(cyc);
  return safe;
}
function hamPickBest(safe, snake, dir, food, path, pos, N) {
  const n = path.length;
  const foodIdx = pos.get(food[0] + ',' + food[1]);
  let best = null, bestD = 1e9;
  for (const a of safe) {
    const r = hamSimulate(snake, dir, a, food, N);
    if (r.dead) continue;
    const d = (((foodIdx - pos.get(r.ns[0][0] + ',' + r.ns[0][1])) % n) + n) % n;
    if (d < bestD) { best = a; bestD = d; }
  }
  return best === null ? hamCycleAction(snake, dir, path, pos) : best;
}
const HAM_PATH = hamBuildCycle(N);
const HAM_POS = new Map();
HAM_PATH.forEach((c, i) => HAM_POS.set(c[0] + ',' + c[1], i));
const HAM_PATH_DIR = hamCycleDirs(HAM_PATH);
function hamExtra(snakeArr, food) {
  const head = snakeArr[0];
  const tail = snakeArr[snakeArr.length - 1];
  const n = HAM_PATH.length;
  if (!food) return [0, 0];
  const hk = HAM_POS.get(head[0] + ',' + head[1]);
  const df = (((HAM_POS.get(food[0] + ',' + food[1]) - hk) % n) + n) % n / n;
  const dt = (((HAM_POS.get(tail[0] + ',' + tail[1]) - hk) % n) + n) % n / n;
  return [df, dt];
}
function aiGetState(snakeArr, dir, food) {
  const hx = snakeArr[0][0], hy = snakeArr[0][1];
  const body = new Set(snakeArr.map(p => p[0] + ',' + p[1]));
  const TURN = {
    up: { 0: 'up', 1: 'left', 2: 'right' },
    down: { 0: 'down', 1: 'right', 2: 'left' },
    left: { 0: 'left', 1: 'down', 2: 'up' },
    right: { 0: 'right', 1: 'up', 2: 'down' },
  };
  const dirIdx = { up: 0, down: 1, left: 2, right: 3 };
  const danger = d => {
    const [dx, dy] = HAM_DIRS[d];
    const nx = hx + dx, ny = hy + dy;
    return (nx < 0 || nx >= N || ny < 0 || ny >= N || body.has(nx + ',' + ny)) ? 1 : 0;
  };
  const safetyLen = d => {
    const [dx, dy] = HAM_DIRS[d];
    let x = hx, y = hy, steps = 0;
    for (;;) { x += dx; y += dy; if (x < 0 || x >= N || y < 0 || y >= N || body.has(x + ',' + y)) break; steps++; }
    return steps / N;
  };
  const wallDist = d => {
    const [dx, dy] = HAM_DIRS[d];
    let x = hx, y = hy, dist = 0;
    for (;;) { x += dx; y += dy; if (x < 0 || x >= N || y < 0 || y >= N) break; dist++; }
    return dist / N;
  };
  const fwd = dir, left = TURN[dir][1], right = TURN[dir][2];
  const isAhead = d => {
    const [dx, dy] = HAM_DIRS[d];
    let nx = hx, ny = hy;
    for (;;) {
      nx += dx; ny += dy;
      if (nx < 0 || nx >= N || ny < 0 || ny >= N) return false;
      if (nx === food[0] && ny === food[1]) return true;
      if (body.has(nx + ',' + ny)) return false;
    }
  };
  const foodRel = [isAhead(fwd) ? 1 : 0, isAhead(left) ? 1 : 0, isAhead(right) ? 1 : 0];
  foodRel.push(foodRel[0] || foodRel[1] || foodRel[2] ? 0 : 1);
  const dirOnehot = [0, 0, 0, 0]; dirOnehot[dirIdx[dir]] = 1;
  const foodDist = (Math.abs(hx - food[0]) + Math.abs(hy - food[1])) / (N + N);
  const freeRatio = (N * N - snakeArr.length) / (N * N);
  let trapped = 0;
  for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
    if (dx === 0 && dy === 0) continue;
    const x = hx + dx, y = hy + dy;
    if (x < 0 || x >= N || y < 0 || y >= N || body.has(x + ',' + y)) trapped++;
  }
  return [
    danger(fwd), danger(left), danger(right), ...foodRel, ...dirOnehot,
    wallDist(fwd), wallDist(left), wallDist(right), foodDist,
    snakeArr.length / (N * N),
    safetyLen(fwd), safetyLen(left), safetyLen(right), freeRatio, trapped / 8,
    ...hamExtra(snakeArr, food),
  ];
}
function aiForward(state, M) {
  const W1flat = M.W1.length > 0 && !Array.isArray(M.W1[0]);
  const W2flat = M.W2p.length > 0 && !Array.isArray(M.W2p[0]);
  const w1 = (i, j) => W1flat ? M.W1[i * M.h + j] : M.W1[i][j];
  const w2 = (j, a) => W2flat ? M.W2p[j * M.na + a] : M.W2p[j][a];
  const h = new Array(M.h).fill(0);
  for (let j = 0; j < M.h; j++) {
    let acc = M.b1[j];
    for (let i = 0; i < state.length; i++) acc += state[i] * w1(i, j);
    h[j] = Math.tanh(acc);
  }
  const logits = new Array(M.na).fill(0);
  for (let a = 0; a < M.na; a++) {
    let acc = M.b2p[a];
    for (let j = 0; j < M.h; j++) acc += h[j] * w2(j, a);
    logits[a] = acc;
  }
  return logits;
}
function aiAct(snakeArr, dir, food, M) {
  const state = aiGetState(snakeArr, dir, food);
  const logits = aiForward(state, M);
  const mx = Math.max(...logits);
  const e = logits.map(v => Math.exp(v - mx));
  const s = e.reduce((a, b) => a + b, 0);
  const probs = e.map(v => v / s);
  const r = Math.random();
  let acc = 0;
  for (let a = 0; a < probs.length; a++) { acc += probs[a]; if (r < acc) return a; }
  return probs.length - 1;
}
function newFood(snake) {
  const occupied = new Set(snake.map(p => p[0] + ',' + p[1]));
  const free = [];
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++)
    if (!occupied.has(x + ',' + y)) free.push([x, y]);
  if (!free.length) return null;
  return free[Math.floor(Math.random() * free.length)];
}
const TURN_A = { up: ['up', 'left', 'right'], down: ['down', 'right', 'left'], left: ['left', 'down', 'up'], right: ['right', 'up', 'down'] };

function playOnce(withFilter) {
  let snake = [[4, 6], [3, 6], [2, 6]];
  let dir = 'right';
  let food = newFood(snake);
  let steps = 0, foods = 0, intervened = 0, win = false;
  while (steps < 8000) {
    let a = aiAct(snake, dir, food, M);
    if (withFilter) {
      const safe = hamSafeMoves(snake, dir, food, HAM_PATH, HAM_POS, HAM_PATH_DIR, N);
      if (!safe.has(a)) { intervened++; a = hamPickBest(safe, snake, dir, food, HAM_PATH, HAM_POS, N); }
    }
    dir = TURN_A[dir][a];
    const [dx, dy] = HAM_DIRS[dir];
    const [hx, hy] = snake[0];
    const nx = hx + dx, ny = hy + dy;
    if (nx < 0 || nx >= N || ny < 0 || ny >= N || snake.some(p => p[0] === nx && p[1] === ny)) break;
    snake.unshift([nx, ny]);
    if (nx === food[0] && ny === food[1]) {
      foods++;
      food = newFood(snake);
      if (!food) { win = true; break; }
    } else snake.pop();
    steps++;
  }
  return { len: snake.length, foods, win, steps, intervened };
}

console.log('文件:', file);
for (const withFilter of [true]) {
  const lens = [], foods = [], wins = [], rates = [];
  for (let i = 0; i < 5; i++) {
    const r = playOnce(withFilter);
    lens.push(r.len); foods.push(r.foods); wins.push(r.win); rates.push(r.intervened / Math.max(r.steps, 1));
  }
  const avg = a => (a.reduce((x, y) => x + y, 0) / a.length).toFixed(1);
  console.log(`带滤波兜底 5 局: 平均身长 ${avg(lens)} 食物 ${avg(foods)} 通关 ${wins.filter(Boolean).length}/5 平均干预率 ${(rates.reduce((a, b) => a + b, 0) / 5 * 100).toFixed(1)}%`);
  console.log('  身长分布:', lens.join(','));
}
