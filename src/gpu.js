// WebGPU device, counted allocations, pipeline cache and timed submits.

/**
 * Opens a WebGPU device with the largest buffer limits the adapter allows.
 * @returns {Promise<Gpu>}
 */
export async function openGpu({ powerPreference = 'high-performance' } = {}) {
  if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference });
  if (!adapter) throw new Error('No WebGPU adapter.');
  const L = adapter.limits;
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: L.maxBufferSize,
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage: Math.min(L.maxStorageBuffersPerShaderStage, 10),
      maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: L.maxComputeWorkgroupSizeX,
    },
  });
  return new Gpu(adapter, device);
}

export class Gpu {
  constructor(adapter, device) {
    this.adapter = adapter;
    this.device = device;
    /** Bytes currently allocated through this object, by tag. */
    this.bytes = new Map();
    this.total = 0;
    this.peak = 0;
    this.pipelines = new Map();
    /** Longest single submit seen, in ms. */
    this.longestSubmit = 0;
    this.lost = null;
    device.lost.then((info) => { this.lost = info; });
  }

  get maxBinding() { return this.device.limits.maxStorageBufferBindingSize; }

  /** Allocates a storage buffer and counts it under `tag`. */
  buffer(size, tag = 'misc', usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
    const padded = Math.max(16, Math.ceil(size / 16) * 16);
    if (padded > this.device.limits.maxBufferSize) throw new Error(`${tag}: ${padded} bytes is over this GPU's buffer limit`);
    const b = this.device.createBuffer({ size: padded, usage });
    b.__tag = tag;
    this.bytes.set(tag, (this.bytes.get(tag) || 0) + padded);
    this.total += padded;
    this.peak = Math.max(this.peak, this.total);
    return b;
  }

  /** Frees a buffer made by `buffer()`. */
  free(b) {
    if (!b || b.__freed) return;
    b.__freed = true;
    this.bytes.set(b.__tag, this.bytes.get(b.__tag) - b.size);
    this.total -= b.size;
    b.destroy();
  }

  uniform(data) {
    const b = this.device.createBuffer({ size: Math.ceil(data.byteLength / 16) * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(b, 0, data);
    return b;
  }

  /** Compiles (once) a compute pipeline for WGSL source. */
  pipeline(code, entry = 'main') {
    const key = entry + '\n' + code;
    let p = this.pipelines.get(key);
    if (!p) {
      const module = this.device.createShaderModule({ code });
      p = this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: entry } });
      this.pipelines.set(key, p);
    }
    return p;
  }

  /** Records one dispatch: `bindings` is an array of buffers bound at 0..n in order. */
  dispatch(enc, code, bindings, groups) {
    const p = this.pipeline(code);
    const bg = this.device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: bindings.map((b, i) => ({ binding: i, resource: b.buffer ? b : { buffer: b } })),
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(p);
    pass.setBindGroup(0, bg);
    const [x, y = 1, z = 1] = Array.isArray(groups) ? groups : [groups];
    pass.dispatchWorkgroups(x, y, z);
    pass.end();
  }

  /** Submits `record(encoder)` as one command buffer and waits for the GPU, timing it. */
  async run(record) {
    const enc = this.device.createCommandEncoder();
    record(enc);
    const t0 = performance.now();
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    const ms = performance.now() - t0;
    this.longestSubmit = Math.max(this.longestSubmit, ms);
    if (this.lost) throw new Error(`GPU lost: ${this.lost.message || this.lost.reason}`);
    return ms;
  }

  /** Copies a buffer back to the CPU. */
  async read(buf, size = buf.size, offset = 0) {
    const staging = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, offset, staging, 0, size);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return out;
  }

  /** Memory report in MB by tag. */
  report() {
    const mb = (n) => Math.round(n / 1048576);
    return { totalMB: mb(this.total), peakMB: mb(this.peak), byTag: Object.fromEntries([...this.bytes].map(([k, v]) => [k, mb(v)])), longestSubmitMs: Math.round(this.longestSubmit) };
  }
}
