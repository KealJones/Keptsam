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

/** Writes K and V rows [T, 4, 128] into ring caches of `slots` slots at position pos0 + t. */
export const kvWrite = /* wgsl */ `
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
  for (var i = lid.x; i < 512u; i += 128u) {
    KC[slot * 512u + i] = K[t * 512u + i];
    VC[slot * 512u + i] = Vv[t * 512u + i];
  }
}`;

// Position held by cache slot j once positions up to maxPos are written (ring of 'slots').
const SLOTPOS = /* wgsl */ `
fn slotPos(j: u32, maxPos: u32, slots: u32) -> i32 {
  let back = (maxPos + slots - (j % slots)) % slots;
  return i32(maxPos) - i32(back);
}`;

/** S[t·16+h, j] = q·k / √128 with causal and window masks (−inf when hidden). */
export const attnScores = /* wgsl */ `
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
  let t = th / 16u; let h = th % 16u;
  let qp = i32(p.pos0 + t);
  let kp = slotPos(j, p.pos0 + p.T - 1u, p.slots);
  var s = -3.0e38;
  if (kp >= 0 && kp <= qp && u32(qp - kp) < p.window) {
    let qb = th * 128u; let kb = j * 512u + (h / 4u) * 128u;
    var acc = 0.0;
    for (var d = 0u; d < 128u; d++) { acc += Q[qb + d] * KC[kb + d]; }
    s = acc * 0.08838834764831845;
  }
  S[th * p.sStride + j] = s;
}`;

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

/** O[t, h, :] = sigmoid(G[t, h]) · Σ_j P[t·16+h, j] · V[j, h/4, :]. */
export const attnOut = /* wgsl */ `
struct P { L: u32, sStride: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read> VC: array<f32>;
@group(0) @binding(3) var<storage, read> G: array<f32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let th = wid.x; let h = th % 16u; let d = lid.x;
  var acc = 0.0;
  for (var j = 0u; j < p.L; j++) {
    let w = S[th * p.sStride + j];
    if (w != 0.0) { acc += w * VC[j * 512u + (h / 4u) * 128u + d]; }
  }
  let g = 1.0 / (1.0 + exp(-G[th]));
  O[th * 128u + d] = acc * g;
}`;

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

/** Non-causal scores for a block of queries: S[(q·H + h), j] = Q[q0+q, h] · K[j, h / (H/HK)] · scale. */
export const encScores = /* wgsl */ `
struct P { N: u32, q0: u32, H: u32, HK: u32, D: u32, scale: f32, pad0: u32, pad1: u32 }
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
