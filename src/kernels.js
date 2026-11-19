// WGSL kernels. Activations are f32. Quantized weights are u32 words (see src/quant.js for layouts).
// No `shader-f16`: halves are unpacked with unpack2x16float.

/** Max tokens one matmul dispatch handles per weight pass; longer batches use more dispatches along z. */
export const MAXT = 8;

const Q6K = /* wgsl */ `
fn byteAt(base: u32, i: u32) -> u32 { return (W[base + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xffu; }
// One value of a padded Q6_K block (53 words) starting at word 'base'.
fn q6k(base: u32, idx: u32) -> f32 {
  let half = idx >> 7u; let l2 = idx & 127u; let quarter = l2 >> 5u; let l = l2 & 31u;
  let qlb = byteAt(base, half * 64u + l + (quarter & 1u) * 32u);
  let lo = select(qlb & 15u, qlb >> 4u, quarter >= 2u);
  let hi = (byteAt(base, 128u + half * 32u + l) >> (quarter * 2u)) & 3u;
  let q = f32(i32(lo | (hi << 4u)) - 32);
  let sb = byteAt(base, 192u + half * 8u + (l >> 4u) + quarter * 2u);
  let sc = f32(select(i32(sb), i32(sb) - 256, sb >= 128u));
  let d = unpack2x16float(W[base + 52u]).x;
  return d * sc * q;
}`;

/** Y[t, r] = W[r, :] · X[t, :] for a Q4_0 weight. One workgroup per row, `mt` tokens per pass (1 for decode). */
export const matmulQ4_0 = (mt = MAXT) => /* wgsl */ `
struct P { rows: u32, K: u32, T: u32, dOff: u32, yStride: u32, yOff: u32, xStride: u32, accumulate: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
const WG = 64u;
const MT = ${mt}u;
var<workgroup> red: array<f32, ${64 * mt}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.y * 32768u + wid.x;
  let t0 = wid.z * MT;
  if (row >= p.rows) { return; }
  let nT = min(MT, p.T - t0);
  let nb = p.K / 32u;
  var acc: array<f32, ${mt}>;
  for (var b = lid.x; b < nb; b += WG) {
    let blk = row * nb + b;
    let dw = W[p.dOff + (blk >> 1u)];
    let d = select(unpack2x16float(dw).x, unpack2x16float(dw).y, (blk & 1u) == 1u);
    for (var w = 0u; w < 4u; w++) {
      let q = W[blk * 4u + w];
      for (var j = 0u; j < 4u; j++) {
        let byte = (q >> (j * 8u)) & 0xffu;
        let lo = (f32(byte & 15u) - 8.0) * d;
        let hi = (f32(byte >> 4u) - 8.0) * d;
        let k = b * 32u + w * 4u + j;
        for (var t = 0u; t < nT; t++) {
          let xb = (t0 + t) * p.xStride + k;
          acc[t] += lo * X[xb] + hi * X[xb + 16u];
        }
      }
    }
  }
  for (var t = 0u; t < MT; t++) { red[t * WG + lid.x] = acc[t]; }
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s >>= 1u) {
    if (lid.x < s) { for (var t = 0u; t < MT; t++) { red[t * WG + lid.x] += red[t * WG + lid.x + s]; } }
    workgroupBarrier();
  }
  if (lid.x < nT) {
    let i = (t0 + lid.x) * p.yStride + p.yOff + row;
    let v = red[lid.x * WG];
    Y[i] = select(v, Y[i] + v, p.accumulate == 1u);
  }
}`;

/** Y[r] = W[r, :] · X for a padded Q6_K weight, one token (the output head). */
export const matmulQ6_K = /* wgsl */ `
struct P { rows: u32, K: u32, xOff: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
${Q6K}
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.y * 32768u + wid.x;
  if (row >= p.rows) { return; }
  let nb = p.K / 256u;
  var acc = 0.0;
  for (var k = lid.x; k < p.K; k += 64u) {
    let base = (row * nb + k / 256u) * 53u;
    acc += q6k(base, k & 255u) * X[p.xOff + k];
  }
  red[lid.x] = acc;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) { if (lid.x < s) { red[lid.x] += red[lid.x + s]; } workgroupBarrier(); }
  if (lid.x == 0u) { Y[row] = red[0]; }
}`;

/** X[t, :] = row tokens[t] of a padded Q6_K table. */
export const embedQ6_K = /* wgsl */ `
struct P { K: u32, T: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> tok: array<u32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;
${Q6K}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  let nb = p.K / 256u;
  for (var k = lid.x; k < p.K; k += 256u) {
    X[t * p.K + k] = q6k((tok[t] * nb + k / 256u) * 53u, k & 255u);
  }
}`;

/** X[t, :] = row tok[t] of a Q4_0 embedding table (K wide), dequantized. Scales start at word dOff. */
export const embedQ4_0 = /* wgsl */ `
struct P { K: u32, dOff: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> tok: array<u32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  let nb = p.K / 32u;
  for (var k = lid.x; k < p.K; k += 256u) {
    let blk = tok[t] * nb + k / 32u;
    let dw = W[p.dOff + (blk >> 1u)];
    let d = select(unpack2x16float(dw).x, unpack2x16float(dw).y, (blk & 1u) == 1u);
    let j = k % 32u; let jj = j % 16u;
    let byte = (W[blk * 4u + jj / 4u] >> ((jj % 4u) * 8u)) & 0xffu;
    let q = select(byte & 15u, byte >> 4u, j >= 16u);
    X[t * p.K + k] = (f32(q) - 8.0) * d;
  }
}`;

/** RMSNorm over rows of length N. mode 0: Y = norm(X)·w. mode 1: Y += norm(X)·w (residual add). */
export const rmsnorm = /* wgsl */ `
struct P { N: u32, mode: u32, eps: f32, wMod: u32, xOff: u32, yOff: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let xb = p.xOff + wid.x * p.N;
  let yb = p.yOff + wid.x * p.N;
  var s = 0.0;
  for (var i = lid.x; i < p.N; i += 128u) { let v = X[xb + i]; s += v * v; }
  red[lid.x] = s;
  workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (lid.x < k) { red[lid.x] += red[lid.x + k]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / f32(p.N) + p.eps);
  for (var i = lid.x; i < p.N; i += 128u) {
    let v = X[xb + i] * inv * w[i % p.wMod];
    if (p.mode == 1u) { Y[yb + i] += v; } else { Y[yb + i] = v; }
  }
}`;

/** NeoX RoPE in place on [T, H, 128]; θ_i = pos · base^(-2i/128) / factor_i. */
export const rope = /* wgsl */ `
struct P { H: u32, pos0: u32, base: f32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> V: array<f32>;
@group(0) @binding(2) var<storage, read> factors: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x / p.H;
  let base = wid.x * 128u;
  let i = lid.x;
  let theta = f32(p.pos0 + t) * pow(p.base, -2.0 * f32(i) / 128.0) / factors[i];
  let c = cos(theta); let s = sin(theta);
  let a = V[base + i]; let b = V[base + i + 64u];
  V[base + i] = a * c - b * s;
  V[base + i + 64u] = a * s + b * c;
}`;

/** Writes K and V rows [T, KVH, 128] into ring caches of `slots` slots at position pos0 + t. */
export const kvWriteFor = (KVH) => /* wgsl */ `
struct P { pos0: u32, slots: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> Vv: array<f32>;
@group(0) @binding(3) var<storage, read_write> KC: array<f32>;
@group(0) @binding(4) var<storage, read_write> VC: array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  let slot = (p.pos0 + t) % p.slots;
  for (var i = lid.x; i < ${KVH * 128}u; i += 128u) {
    KC[slot * ${KVH * 128}u + i] = K[t * ${KVH * 128}u + i];
    VC[slot * ${KVH * 128}u + i] = Vv[t * ${KVH * 128}u + i];
  }
}`;
export const kvWrite = kvWriteFor(4);

// Position held by cache slot j once positions up to maxPos are written (ring of 'slots').
const SLOTPOS = /* wgsl */ `
fn slotPos(j: u32, maxPos: u32, slots: u32) -> i32 {
  let back = (maxPos + slots - (j % slots)) % slots;
  return i32(maxPos) - i32(back);
}`;

/** S[t·H+h, j] = q·k / √128 with causal and window masks (−inf when hidden); H query heads share KVH KV heads. */
export const attnScoresFor = (H, KVH) => /* wgsl */ `
struct P { pos0: u32, T: u32, slots: u32, L: u32, window: u32, sStride: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> KC: array<f32>;
@group(0) @binding(3) var<storage, read_write> S: array<f32>;
${SLOTPOS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  let th = gid.y;
  if (j >= p.L) { return; }
  let t = th / ${H}u; let h = th % ${H}u;
  let qp = i32(p.pos0 + t);
  let kp = slotPos(j, p.pos0 + p.T - 1u, p.slots);
  var s = -3.0e38;
  if (kp >= 0 && kp <= qp && u32(qp - kp) < p.window) {
    let qb = th * 128u; let kb = j * ${KVH * 128}u + (h / ${H / KVH}u) * 128u;
    var acc = 0.0;
    for (var d = 0u; d < 128u; d++) { acc += Q[qb + d] * KC[kb + d]; }
    s = acc * 0.08838834764831845;
  }
  S[th * p.sStride + j] = s;
}`;
export const attnScores = attnScoresFor(16, 4);

/** Softmax in place over rows of length L. */
export const softmax = /* wgsl */ `
struct P { L: u32, stride: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> S: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let b = wid.x * p.stride;
  var m = -3.0e38;
  for (var j = lid.x; j < p.L; j += 256u) { m = max(m, S[b + j]); }
  red[lid.x] = m; workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (lid.x < k) { red[lid.x] = max(red[lid.x], red[lid.x + k]); } workgroupBarrier(); }
  m = red[0]; workgroupBarrier();
  var s = 0.0;
  for (var j = lid.x; j < p.L; j += 256u) { let e = select(exp(S[b + j] - m), 0.0, S[b + j] <= -1.0e38); S[b + j] = e; s += e; }
  red[lid.x] = s; workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (lid.x < k) { red[lid.x] += red[lid.x + k]; } workgroupBarrier(); }
  let inv = 1.0 / red[0];
  for (var j = lid.x; j < p.L; j += 256u) { S[b + j] *= inv; }
}`;

/** O[t, h, :] = sigmoid(G[t, h]) · Σ_j P[t·H+h, j] · V[j, h/(H/KVH), :]; without `gated`, no G and no sigmoid. */
export const attnOutFor = (H, KVH, gated = true) => /* wgsl */ `
struct P { L: u32, sStride: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read> VC: array<f32>;
${gated ? `@group(0) @binding(3) var<storage, read> G: array<f32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;` : '@group(0) @binding(3) var<storage, read_write> O: array<f32>;'}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let th = wid.x; let h = th % ${H}u; let d = lid.x;
  var acc = 0.0;
  for (var j = 0u; j < p.L; j++) {
    let w = S[th * p.sStride + j];
    if (w != 0.0) { acc += w * VC[j * ${KVH * 128}u + (h / ${H / KVH}u) * 128u + d]; }
  }
${gated ? `  let g = 1.0 / (1.0 + exp(-G[th]));
  O[th * 128u + d] = acc * g;` : '  O[th * 128u + d] = acc;'}
}`;
export const attnOut = attnOutFor(16, 4);

/** LFM short conv over T tokens: Y = C ⊙ conv7(B ⊙ X), keeping the last 6 inputs per channel as state. */
export const shortconv = /* wgsl */ `
struct P { T: u32, N: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> BCX: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> state: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ch = gid.x;
  if (ch >= p.N) { return; }
  var win: array<f32, 7>;
  for (var k = 0u; k < 6u; k++) { win[k] = state[k * p.N + ch]; }
  for (var t = 0u; t < p.T; t++) {
    let row = t * 3u * p.N;
    win[6] = BCX[row + ch] * BCX[row + 2u * p.N + ch];
    var acc = 0.0;
    for (var k = 0u; k < 7u; k++) { acc += win[k] * w[ch * 7u + k]; }
    Y[t * p.N + ch] = BCX[row + p.N + ch] * acc;
    for (var k = 0u; k < 6u; k++) { win[k] = win[k + 1u]; }
  }
  for (var k = 0u; k < 6u; k++) { state[k * p.N + ch] = win[k]; }
}`;

/** G = silu(G) ⊙ U, elementwise over n values. */
export const swiglu = /* wgsl */ `
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> G: array<f32>;
@group(0) @binding(2) var<storage, read> U: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * 65535u * 256u + gid.x;
  if (i >= p.n) { return; }
  let g = G[i];
  G[i] = g / (1.0 + exp(-g)) * U[i];
}`;

// ───────────── Shared by the Omni senses (encoders run all positions at once, non-causal) ─────────────

/** Y[t, r] = W[r, :] · X[t, :] (+ b[r]) for a Q8_0 weight. One workgroup per row, `mt` tokens per pass. */
export const matmulQ8_0 = (mt = MAXT) => /* wgsl */ `
struct P { rows: u32, K: u32, T: u32, dOff: u32, bias: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> B: array<f32>;
const WG = 64u;
const MT = ${mt}u;
var<workgroup> red: array<f32, ${64 * mt}>;
fn s8(w: u32, j: u32) -> f32 { let b = (w >> (j * 8u)) & 0xffu; return f32(select(i32(b), i32(b) - 256, b >= 128u)); }
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.y * 32768u + wid.x;
  let t0 = wid.z * MT;
  if (row >= p.rows) { return; }
  let nT = min(MT, p.T - t0);
  let nb = p.K / 32u;
  var acc: array<f32, ${mt}>;
  for (var b = lid.x; b < nb; b += WG) {
    let blk = row * nb + b;
    let dw = W[p.dOff + (blk >> 1u)];
    let d = select(unpack2x16float(dw).x, unpack2x16float(dw).y, (blk & 1u) == 1u);
    for (var w = 0u; w < 8u; w++) {
      let q = W[blk * 8u + w];
      for (var j = 0u; j < 4u; j++) {
        let v = s8(q, j) * d;
        let k = b * 32u + w * 4u + j;
        for (var t = 0u; t < nT; t++) { acc[t] += v * X[(t0 + t) * p.K + k]; }
      }
    }
  }
  for (var t = 0u; t < MT; t++) { red[t * WG + lid.x] = acc[t]; }
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s >>= 1u) {
    if (lid.x < s) { for (var t = 0u; t < MT; t++) { red[t * WG + lid.x] += red[t * WG + lid.x + s]; } }
    workgroupBarrier();
  }
  if (lid.x < nT) {
    var v = red[lid.x * WG];
    if (p.bias == 1u) { v += B[row]; }
    Y[(t0 + lid.x) * p.rows + row] = v;
  }
}`;

/** Y[t, r] = W[r, :] · X[t, :] (+ b[r]) for an f32 weight. */
export const matmulF32 = (mt = MAXT) => /* wgsl */ `
struct P { rows: u32, K: u32, T: u32, bias: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> B: array<f32>;
const WG = 64u;
const MT = ${mt}u;
var<workgroup> red: array<f32, ${64 * mt}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.y * 32768u + wid.x;
  let t0 = wid.z * MT;
  if (row >= p.rows) { return; }
  let nT = min(MT, p.T - t0);
  var acc: array<f32, ${mt}>;
  for (var k = lid.x; k < p.K; k += WG) {
    let w = W[row * p.K + k];
    for (var t = 0u; t < nT; t++) { acc[t] += w * X[(t0 + t) * p.K + k]; }
  }
  for (var t = 0u; t < MT; t++) { red[t * WG + lid.x] = acc[t]; }
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s >>= 1u) {
    if (lid.x < s) { for (var t = 0u; t < MT; t++) { red[t * WG + lid.x] += red[t * WG + lid.x + s]; } }
    workgroupBarrier();
  }
  if (lid.x < nT) {
    var v = red[lid.x * WG];
    if (p.bias == 1u) { v += B[row]; }
    Y[(t0 + lid.x) * p.rows + row] = v;
  }
}`;

/** LayerNorm with bias over rows of length N. */
export const layernorm = /* wgsl */ `
struct P { N: u32, eps: f32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let base = wid.x * p.N;
  var s = 0.0;
  for (var i = lid.x; i < p.N; i += 128u) { s += X[base + i]; }
  red[lid.x] = s; workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (lid.x < k) { red[lid.x] += red[lid.x + k]; } workgroupBarrier(); }
  let mean = red[0] / f32(p.N); workgroupBarrier();
  var v = 0.0;
  for (var i = lid.x; i < p.N; i += 128u) { let d = X[base + i] - mean; v += d * d; }
  red[lid.x] = v; workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (lid.x < k) { red[lid.x] += red[lid.x + k]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / f32(p.N) + p.eps);
  for (var i = lid.x; i < p.N; i += 128u) { Y[base + i] = (X[base + i] - mean) * inv * w[i] + b[i]; }
}`;

/** In place: X = gelu(X) with erf (HF "gelu"). */
export const geluErf = /* wgsl */ `
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
// erf with |error| < 1.5e-7 (Abramowitz & Stegun 7.1.26)
fn erf(x: f32) -> f32 {
  let s = sign(x); let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return s * y;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * 65535u * 256u + gid.x;
  if (i >= p.n) { return; }
  let x = X[i];
  X[i] = 0.5 * x * (1.0 + erf(x * 0.7071067811865476));
}`;

/** Y = X + R elementwise (residual add into Y). */
export const addInto = /* wgsl */ `
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * 65535u * 256u + gid.x;
  if (i < p.n) { Y[i] += X[i]; }
}`;

/** LFM3-VL 2D RoPE in place on [N, H, 64]: pairs (i, i+32); i < 16 rotate by row y, i ≥ 16 by column x. */
export const rope2d = /* wgsl */ `
struct P { H: u32, pw: u32, base: f32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> V: array<f32>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wid.x / p.H;
  let base = wid.x * 64u;
  let i = lid.x;
  let y = f32(n / p.pw); let x = f32(n % p.pw);
  let k = select(i, i - 16u, i >= 16u);
  let pos = select(y, x, i >= 16u);
  let theta = pos * pow(p.base, -2.0 * f32(k) / 32.0);
  let c = cos(theta); let s = sin(theta);
  let a = V[base + i]; let b = V[base + i + 32u];
  V[base + i] = a * c - b * s;
  V[base + i + 32u] = a * s + b * c;
}`;

/**
 * Scores for a block of queries: S[(q·H + h), j] = Q[q0+q, h] · K[j, h / (H/HK)] · scale.
 * window 0: every key (non-causal). window w > 0: causal, keys within the last w positions.
 */
export const encScores = /* wgsl */ `
struct P { N: u32, q0: u32, H: u32, HK: u32, D: u32, scale: f32, window: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read_write> S: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x; let qh = gid.y;
  if (j >= p.N) { return; }
  let q = qh / p.H; let h = qh % p.H;
  let qb = ((p.q0 + q) * p.H + h) * p.D;
  let kb = (j * p.HK + h / (p.H / p.HK)) * p.D;
  let qa = p.q0 + q;
  if (p.window > 0u && (j > qa || qa - j >= p.window)) { S[qh * p.N + j] = -3.0e38; return; }
  var acc = 0.0;
  for (var d = 0u; d < p.D; d++) { acc += Q[qb + d] * K[kb + d]; }
  S[qh * p.N + j] = acc * p.scale;
}`;

/** O[q0+q, h, :] = Σ_j S[(q·H + h), j] · V[j, h / (H/HK), :]. One workgroup per (q, h), D threads. */
export const encOut = /* wgsl */ `
struct P { N: u32, q0: u32, H: u32, HK: u32, D: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let qh = wid.x; let q = qh / p.H; let h = qh % p.H;
  for (var d = lid.x; d < p.D; d += 64u) {
    var acc = 0.0;
    for (var j = 0u; j < p.N; j++) { acc += S[qh * p.N + j] * V[(j * p.HK + h / (p.H / p.HK)) * p.D + d]; }
    O[((p.q0 + q) * p.H + h) * p.D + d] = acc;
  }
}`;

/** LFM3-VL conv block core: Y[n, c] = middle · dwconv3x3(first ⊙ third) over the pw × ph patch grid (zero pad). */
export const dwconv3 = /* wgsl */ `
struct P { pw: u32, ph: u32, C: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x; let n = gid.y;
  if (c >= p.C) { return; }
  let y = i32(n / p.pw); let x = i32(n % p.pw);
  var acc = 0.0;
  for (var ky = 0; ky < 3; ky++) {
    for (var kx = 0; kx < 3; kx++) {
      let yy = y + ky - 1; let xx = x + kx - 1;
      if (yy >= 0 && xx >= 0 && yy < i32(p.ph) && xx < i32(p.pw)) {
        let m = u32(yy) * p.pw + u32(xx);
        let g = X[m * 3u * p.C + c] * X[m * 3u * p.C + 2u * p.C + c];
        acc += g * w[c * 9u + u32(ky) * 3u + u32(kx)];
      }
    }
  }
  Y[n * p.C + c] = X[n * 3u * p.C + p.C + c] * acc;
}`;

/** 2×2 patch merge: out[t] = [patch(2gx,2gy), patch(2gx+1,2gy), patch(2gx,2gy+1), patch(2gx+1,2gy+1)]. */
export const merge2x2 = /* wgsl */ `
struct P { pw: u32, C: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  let gw = p.pw / 2u;
  let gx = t % gw; let gy = t / gw;
  for (var k = lid.x; k < 4u * p.C; k += 256u) {
    let sub = k / p.C; let c = k % p.C;
    let px = 2u * gx + (sub & 1u); let py = 2u * gy + (sub >> 1u);
    Y[t * 4u * p.C + k] = X[(py * p.pw + px) * p.C + c];
  }
}`;

/**
 * Tiled Y[t, r] = W[r, :] · X[t, :] for a Q4_0 weight and many tokens (prompt reading).
 * Each workgroup computes 64 rows × 32 tokens through shared memory, so each weight block is read and
 * dequantized once per 32 tokens; each thread accumulates 4 rows × 2 tokens.
 * Same parameters as matmulQ4_0.
 */
export const gemmQ4_0 = /* wgsl */ `
struct P { rows: u32, K: u32, T: u32, dOff: u32, yStride: u32, yOff: u32, xStride: u32, accumulate: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> Ws: array<f32, 2048>; // [32 k][64 rows]
var<workgroup> Xs: array<f32, 1024>; // [32 k][32 tokens]
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let row0 = wid.x * 64u;
  let tok0 = wid.y * 32u;
  let nb = p.K / 32u;
  let tr = li / 16u; let tc = li % 16u;
  var acc: array<f32, 8>;
  // loader roles
  let lr = li / 4u; let lq = li % 4u;
  let lrow = row0 + lr;
  for (var b = 0u; b < nb; b++) {
    // 64 rows × 1 block: each thread dequantizes one word (8 values) of one row's block.
    if (lrow < p.rows) {
      let blk = lrow * nb + b;
      let dw = W[p.dOff + (blk >> 1u)];
      let d = select(unpack2x16float(dw).x, unpack2x16float(dw).y, (blk & 1u) == 1u);
      let q = W[blk * 4u + lq];
      for (var j = 0u; j < 4u; j++) {
        let byte = (q >> (j * 8u)) & 0xffu;
        let k = lq * 4u + j;
        Ws[k * 64u + lr] = (f32(byte & 15u) - 8.0) * d;
        Ws[(k + 16u) * 64u + lr] = (f32(byte >> 4u) - 8.0) * d;
      }
    } else {
      for (var j = 0u; j < 4u; j++) { Ws[(lq * 4u + j) * 64u + lr] = 0.0; Ws[(lq * 4u + j + 16u) * 64u + lr] = 0.0; }
    }
    // 32 tokens × 32 values of X.
    for (var m = 0u; m < 4u; m++) {
      let idx = li + 256u * m;
      let t = idx / 32u; let k = idx % 32u;
      let tok = tok0 + t;
      Xs[k * 32u + t] = select(0.0, X[tok * p.xStride + b * 32u + k], tok < p.T);
    }
    workgroupBarrier();
    for (var k = 0u; k < 32u; k++) {
      let x0 = Xs[k * 32u + tc]; let x1 = Xs[k * 32u + tc + 16u];
      for (var i = 0u; i < 4u; i++) {
        let w = Ws[k * 64u + tr + 16u * i];
        acc[i * 2u] += w * x0;
        acc[i * 2u + 1u] += w * x1;
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let row = row0 + tr + 16u * i;
    if (row < p.rows) {
      for (var j = 0u; j < 2u; j++) {
        let tok = tok0 + tc + 16u * j;
        if (tok < p.T) { Y[tok * p.yStride + p.yOff + row] = acc[i * 2u + j]; }
      }
    }
  }
}`;

/**
 * Top 16 of each 256-value slice of the logits: C[(slice·16 + i)·2] = value bits, C[… + 1] = token id.
 * Reading back these candidates (8 KB per 1,000 slices) replaces reading back every logit; the true top-k is in
 * them unless more than 16 of it share one slice of 256 neighbouring token ids.
 */
export const topk16 = /* wgsl */ `
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> L: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<u32>;
var<workgroup> v: array<f32, 256>;
var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let i = wid.x * 256u + lid.x;
  v[lid.x] = select(-3.0e38, L[i], i < p.n);
  workgroupBarrier();
  for (var r = 0u; r < 16u; r++) {
    rv[lid.x] = v[lid.x]; ri[lid.x] = lid.x;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
      if (lid.x < s && rv[lid.x + s] > rv[lid.x]) { rv[lid.x] = rv[lid.x + s]; ri[lid.x] = ri[lid.x + s]; }
      workgroupBarrier();
    }
    let best = ri[0];
    if (lid.x == 0u) {
      C[(wid.x * 16u + r) * 2u] = bitcast<u32>(rv[0]);
      C[(wid.x * 16u + r) * 2u + 1u] = wid.x * 256u + best;
    }
    if (lid.x == best) { v[lid.x] = -3.0e38; }
    workgroupBarrier();
  }
}`;

/**
 * Single-token y = W·x for Q4_0: 8 rows per 256-thread workgroup (32 threads per row, striding over blocks),
 * so far fewer, fuller workgroups than one per row. Same parameters as matmulQ4_0 (T must be 1).
 */
export const matvecQ4_0 = /* wgsl */ `
struct P { rows: u32, K: u32, T: u32, dOff: u32, yStride: u32, yOff: u32, xStride: u32, accumulate: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let g = li / 32u; let lane = li % 32u;
  let row = (wid.y * 4096u + wid.x) * 8u + g;
  let nb = p.K / 32u;
  var acc = 0.0;
  if (row < p.rows) {
    for (var b = lane; b < nb; b += 32u) {
      let blk = row * nb + b;
      let dw = W[p.dOff + (blk >> 1u)];
      let d = select(unpack2x16float(dw).x, unpack2x16float(dw).y, (blk & 1u) == 1u);
      let xb = b * 32u;
      var s = 0.0;
      for (var w = 0u; w < 4u; w++) {
        let q = W[blk * 4u + w];
        for (var j = 0u; j < 4u; j++) {
          let byte = (q >> (j * 8u)) & 0xffu;
          let k = xb + w * 4u + j;
          s += (f32(byte & 15u) - 8.0) * X[k] + (f32(byte >> 4u) - 8.0) * X[k + 16u];
        }
      }
      acc += s * d;
    }
  }
  red[li] = acc;
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  if (lane == 0u && row < p.rows) { Y[p.yOff + row] = red[li]; }
}`;


// ───────────── LFM3 audio encoder ─────────────

/**
 * Stride-2 3×3 conv over [channel][freq][time] (time fastest): one zero column before the clip and one after it
 * in time, one zero row on each side in frequency. depthwise: channel c reads input channel c; otherwise all
 * output channels read the single input channel. Y = silu?(conv + bias).
 */
export const convDown = /* wgsl */ `
struct P { Tin: u32, Fin: u32, Tout: u32, Fout: u32, C: u32, depthwise: u32, silu: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let to = gid.x; let fo = gid.y % p.Fout; let c = gid.y / p.Fout;
  if (to >= p.Tout || c >= p.C) { return; }
  let ci = select(0u, c, p.depthwise == 1u);
  var acc = b[c];
  for (var kf = 0u; kf < 3u; kf++) {
    let fi = i32(2u * fo + kf) - 1;
    if (fi < 0 || fi >= i32(p.Fin)) { continue; }
    for (var kt = 0u; kt < 3u; kt++) {
      let ti = i32(2u * to + kt) - 1;
      if (ti < 0 || ti >= i32(p.Tin)) { continue; }
      acc += w[c * 9u + kf * 3u + kt] * X[(ci * p.Fin + u32(fi)) * p.Tin + u32(ti)];
    }
  }
  if (p.silu == 1u) { acc = acc / (1.0 + exp(-acc)); }
  Y[(c * p.Fout + fo) * p.Tout + to] = acc;
}`;

/** 1×1 conv over channels for [channel][pixels]: Y[co][q] = silu(Σ_ci w[co, ci] · X[ci][q] + b[co]). */
export const pointwise = /* wgsl */ `
struct P { C: u32, Q: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x; let co = gid.y;
  if (q >= p.Q) { return; }
  var acc = b[co];
  for (var ci = 0u; ci < p.C; ci++) { acc += w[co * p.C + ci] * X[ci * p.Q + q]; }
  Y[co * p.Q + q] = acc / (1.0 + exp(-acc));
}`;

/** [channel][freq][time] → rows per time step: Y[t, c·F + f] = X[c][f][t]. */
export const flattenCFT = /* wgsl */ `
struct P { C: u32, F: u32, T: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  for (var k = lid.x; k < p.C * p.F; k += 256u) { Y[t * p.C * p.F + k] = X[k * p.T + t]; }
}`;

/** Causal short conv of width W over T tokens from a zero state: Y = C ⊙ conv(B ⊙ X). Channels N, BCX rows 3N. */
export const shortconvFresh = (W) => /* wgsl */ `
struct P { T: u32, N: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> BCX: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ch = gid.x; let t = gid.y;
  if (ch >= p.N) { return; }
  var acc = 0.0;
  for (var k = 0u; k < ${W}u; k++) {
    let ti = i32(t) - i32(${W - 1}u) + i32(k);
    if (ti >= 0) {
      let row = u32(ti) * 3u * p.N;
      acc += BCX[row + ch] * BCX[row + 2u * p.N + ch] * w[ch * ${W}u + k];
    }
  }
  Y[t * p.N + ch] = BCX[t * 3u * p.N + p.N + ch] * acc;
}`;

/** NeoX RoPE in place on [T, H, D] at positions pos0 + t: pairs (i, i + D/2), θ_i = pos · base^(-2i/D). */
export const ropeN = /* wgsl */ `
struct P { H: u32, pos0: u32, base: f32, D: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> V: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x / p.H;
  let base = wid.x * p.D;
  let half = p.D / 2u;
  for (var i = lid.x; i < half; i += 64u) {
    let theta = f32(p.pos0 + t) * pow(p.base, -2.0 * f32(i) / f32(p.D));
    let c = cos(theta); let s = sin(theta);
    let a = V[base + i]; let b = V[base + i + half];
    V[base + i] = a * c - b * s;
    V[base + i + half] = a * s + b * c;
  }
}`;

// ───────────── LFM3 speech generation (depthformer + codec) ─────────────

/** Rows idx[i] of a Q8_0 matrix (K columns), dequantized: Y[i, :]. idx are u32 in `I`. */
export const gatherQ8_0 = /* wgsl */ `
struct P { K: u32, dOff: u32, n: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> I: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let i = wid.x;
  let row = I[i];
  let nb = p.K / 32u;
  for (var k = lid.x; k < p.K; k += 64u) {
    let blk = row * nb + k / 32u;
    let dw = W[p.dOff + (blk >> 1u)];
    let d = select(unpack2x16float(dw).x, unpack2x16float(dw).y, (blk & 1u) == 1u);
    let j = k % 32u;
    let byte = (W[blk * 8u + j / 4u] >> ((j % 4u) * 8u)) & 0xffu;
    Y[i * p.K + k] = f32(select(i32(byte), i32(byte) - 256, byte >= 128u)) * d;
  }
}`;

/** Depthformer input: X[i] = P[i] + (mask[i] ? T[i] : M) for n positions of width K (mask as f32 0/1). */
export const depthInput = /* wgsl */ `
struct P { K: u32, n: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Pj: array<f32>;
@group(0) @binding(2) var<storage, read> T: array<f32>;
@group(0) @binding(3) var<storage, read> M: array<f32>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read_write> X: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= p.K * p.n) { return; }
  let i = idx / p.K; let k = idx % p.K;
  X[idx] = Pj[idx] + mask[i] * T[idx] + (1.0 - mask[i]) * M[k];
}`;

/** Y[k] = Σ_i X[i, k] over n rows of width K. */
export const sumRows = /* wgsl */ `
struct P { K: u32, n: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= p.K) { return; }
  var s = 0.0;
  for (var i = 0u; i < p.n; i++) { s += X[i * p.K + k]; }
  Y[k] = s;
}`;

/** RMSNorm with a different weight row per input row: Y[r] = norm(X[r]) · w[r]. */
export const rmsnormRowW = /* wgsl */ `
struct P { N: u32, eps: f32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let b = wid.x * p.N;
  var s = 0.0;
  for (var i = lid.x; i < p.N; i += 128u) { let v = X[b + i]; s += v * v; }
  red[lid.x] = s; workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (lid.x < k) { red[lid.x] += red[lid.x + k]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / f32(p.N) + p.eps);
  for (var i = lid.x; i < p.N; i += 128u) { Y[b + i] = X[b + i] * inv * w[b + i]; }
}`;

/** Per-group matvec on a Q8_0 matrix: row r uses input row r / rowsPerGroup (codebook logits). */
export const matvecQ8Grouped = /* wgsl */ `
struct P { rows: u32, K: u32, dOff: u32, rowsPerGroup: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 256>;
fn s8(w: u32, j: u32) -> f32 { let b = (w >> (j * 8u)) & 0xffu; return f32(select(i32(b), i32(b) - 256, b >= 128u)); }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let g = li / 32u; let lane = li % 32u;
  let row = (wid.y * 4096u + wid.x) * 8u + g;
  let nb = p.K / 32u;
  var acc = 0.0;
  if (row < p.rows) {
    let xb = (row / p.rowsPerGroup) * p.K;
    for (var b = lane; b < nb; b += 32u) {
      let blk = row * nb + b;
      let dw = W[p.dOff + (blk >> 1u)];
      let d = select(unpack2x16float(dw).x, unpack2x16float(dw).y, (blk & 1u) == 1u);
      var s = 0.0;
      for (var w = 0u; w < 8u; w++) {
        let q = W[blk * 8u + w];
        for (var j = 0u; j < 4u; j++) { s += s8(q, j) * X[xb + b * 32u + w * 4u + j]; }
      }
      acc += s * d;
    }
  }
  red[li] = acc;
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) { if (lane < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  if (lane == 0u && row < p.rows) { Y[row] = red[li]; }
}`;

/** In place: O[t, h, :] *= sigmoid(G[t·H + h]) for heads of size D. */
export const gateHeads = /* wgsl */ `
struct P { H: u32, D: u32, n: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> O: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  O[i] *= 1.0 / (1.0 + exp(-G[i / p.D]));
}`;

/** Y[t·R + r, :] = X[t, :]: every row repeated R times (nearest upsampling). */
export const repeatRows = /* wgsl */ `
struct P { K: u32, R: u32, n: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let row = i / p.K; let k = i % p.K;
  Y[i] = X[(row / p.R) * p.K + k];
}`;

/** Copy rows idx[i] of an f32 table (K columns): Y[i, :] = W[idx[i], :]. */
export const gatherF32 = /* wgsl */ `
struct P { K: u32, n: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> I: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n * p.K) { return; }
  Y[i] = W[I[i / p.K] * p.K + i % p.K];
}`;

/**
 * Causal sliding-window attention over a ring KV cache with general head sizes (codec decoder).
 * Scores: S[(t·H + h), j] for query positions pos0 + t, cache slot j holding the latest position ≡ j (mod slots).
 */
export const ringScores = /* wgsl */ `
struct P { pos0: u32, T: u32, slots: u32, L: u32, window: u32, H: u32, HK: u32, D: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> KC: array<f32>;
@group(0) @binding(3) var<storage, read_write> S: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x; let th = gid.y;
  if (j >= p.L) { return; }
  let t = th / p.H; let h = th % p.H;
  let qp = i32(p.pos0 + t);
  let maxPos = p.pos0 + p.T - 1u;
  let back = (maxPos + p.slots - (j % p.slots)) % p.slots;
  let kp = i32(maxPos) - i32(back);
  var s = -3.0e38;
  if (kp >= 0 && kp <= qp && u32(qp - kp) < p.window) {
    let qb = th * p.D; let kb = (j * p.HK + h / (p.H / p.HK)) * p.D;
    var acc = 0.0;
    for (var d = 0u; d < p.D; d++) { acc += Q[qb + d] * KC[kb + d]; }
    s = acc * inverseSqrt(f32(p.D));
  }
  S[th * p.L + j] = s;
}`;

/** O[t, h, :] = Σ_j S[(t·H + h), j] · VC[j, h / (H/HK), :]. */
export const ringOut = /* wgsl */ `
struct P { L: u32, H: u32, HK: u32, D: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read> VC: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let th = wid.x; let h = th % p.H;
  for (var d = lid.x; d < p.D; d += 64u) {
    var acc = 0.0;
    for (var j = 0u; j < p.L; j++) { acc += S[th * p.L + j] * VC[(j * p.HK + h / (p.H / p.HK)) * p.D + d]; }
    O[th * p.D + d] = acc;
  }
}`;

/** Writes T rows of width W into a ring cache of `slots` rows at positions pos0 + t. */
export const ringWrite = /* wgsl */ `
struct P { pos0: u32, slots: u32, W: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  let slot = (p.pos0 + t) % p.slots;
  for (var i = lid.x; i < p.W; i += 64u) { C[slot * p.W + i] = X[t * p.W + i]; }
}`;
