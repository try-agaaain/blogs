// 冒烟测试 v3 HTML 的哈密顿 JS 逻辑（与 src/hamilton.py 对拍）
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
function hamBuildCycle(N) {
  const order = [];
  for (let y = 0; y < N; y++) {
    if (y % 2 === 0) { for (let x = 1; x < N; x++) order.push([x, y]); }
    else { for (let x = N - 1; x >= 1; x--) order.push([x, y]); }
  }
  for (let y = N - 1; y >= 0; y--) order.push([0, y]);
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

// ---- 简单蛇环境模拟（与 Python SnakeEnv 对齐）----
function newFood(snake) {
  const occupied = new Set(snake.map(p => p[0] + ',' + p[1]));
  const free = [];
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    if (!occupied.has(x + ',' + y)) free.push([x, y]);
  }
  if (!free.length) return null;
  return free[Math.floor(Math.random() * free.length)];
}

// 用纯启发式（hamPickBest from safe）玩一局，验证安全滤波正确性
function playOnce(seedless) {
  let snake = [[4, 6], [3, 6], [2, 6]];
  let dir = 'right';
  let food = newFood(snake);
  let steps = 0, foods = 0;
  while (steps < 40000) {
    const safe = hamSafeMoves(snake, dir, food, HAM_PATH, HAM_POS, HAM_PATH_DIR, N);
    const a = hamPickBest(safe, snake, dir, food, HAM_PATH, HAM_POS, N);
    dir = HAM_TURN[dir][a];
    const [dx, dy] = HAM_DIRS[dir];
    const [hx, hy] = snake[0];
    const nx = hx + dx, ny = hy + dy;
    if (nx < 0 || nx >= N || ny < 0 || ny >= N) { console.log('撞墙!', steps); return { len: snake.length, foods }; }
    if (snake.some(p => p[0] === nx && p[1] === ny)) { console.log('撞身!', steps); return { len: snake.length, foods }; }
    snake.unshift([nx, ny]);
    if (nx === food[0] && ny === food[1]) { foods++; food = newFood(snake); if (!food) return { len: snake.length, foods }; }
    else snake.pop();
    steps++;
  }
  return { len: snake.length, foods };
}

const lens = [], foodses = [];
for (let i = 0; i < 8; i++) {
  const r = playOnce();
  lens.push(r.len); foodses.push(r.foods);
}
console.log('JS 启发式 8 局: 平均身长', (lens.reduce((a, b) => a + b, 0) / 8).toFixed(1), '食物', (foodses.reduce((a, b) => a + b, 0) / 8).toFixed(1));
console.log('身长分布:', lens);
console.log('回路长度:', HAM_PATH.length);
