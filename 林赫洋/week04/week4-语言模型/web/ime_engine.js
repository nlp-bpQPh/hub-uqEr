/*
 * ime_engine.js — 纯前端字符级 LSTM 语言模型推理引擎
 *
 * 不依赖任何框架：Embedding 查表 + 2 层 LSTM + Linear，手写矩阵向量乘。
 * 模型权重由 tools/export_ime_model.py 导出成 ime_model.js（fp16 + base64），
 * 本文件负责解码 + 前向 + 拼音切分 + 束搜索解码。
 *
 * 浏览器里：<script src="ime_model.js"></script><script src="ime_engine.js"></script>
 * Node 里（自检用）：global.window = globalThis; require('./ime_model.js'); require('./ime_engine.js')
 */
(function (root) {
  "use strict";

  // ─────────────────────────── 基础工具 ───────────────────────────

  function b64ToBinary(b64) {
    if (typeof atob === "function") return atob(b64);
    return Buffer.from(b64, "base64").toString("binary");
  }

  // IEEE754 半精度 → 单精度
  function halfToFloat(h) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h & 0x7c00) >> 10;
    const f = h & 0x03ff;
    if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + f / 1024);
  }

  function decodeFp16(b64, count) {
    const bin = b64ToBinary(b64);
    if (bin.length !== count * 2) {
      throw new Error(`权重长度不符: 期望 ${count * 2} 字节, 实际 ${bin.length}`);
    }
    const out = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      out[i] = halfToFloat(bin.charCodeAt(i * 2) | (bin.charCodeAt(i * 2 + 1) << 8));
    }
    return out;
  }

  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const tanh = Math.tanh;

  function logSoftmaxArr(logits, temperature) {
    const t = temperature && temperature > 0 ? temperature : 1;
    const n = logits.length;
    let max = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = logits[i] / t;
      if (v > max) max = v;
    }
    let sum = 0;
    for (let i = 0; i < n; i++) sum += Math.exp(logits[i] / t - max);
    const logZ = max + Math.log(sum);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = logits[i] / t - logZ;
    return out;
  }

  // ─────────────────────────── 模型 ───────────────────────────

  class CharLSTM {
    constructor(data) {
      this.data = data;
      const meta = data.meta;
      this.meta = meta;
      this.V = meta.vocab_size;
      this.E = meta.embed_dim;
      this.H = meta.hidden_dim;
      this.L = meta.num_layers;

      this.vocab = data.vocab.split("");
      this.char2id = new Map();
      for (let i = 0; i < this.vocab.length; i++) this.char2id.set(this.vocab[i], i);

      const w = data.weights;
      this.embed = decodeFp16(w.embed, this.V * this.E);
      this.fcW = decodeFp16(w.fc_w, this.V * this.H);
      this.fcB = decodeFp16(w.fc_b, this.V);
      this.layers = w.lstm.map((l) => ({
        in: l.in,
        hidden: l.hidden,
        wIh: decodeFp16(l.weight_ih, 4 * l.hidden * l.in),
        wHh: decodeFp16(l.weight_hh, 4 * l.hidden * l.hidden),
        bias: decodeFp16(l.bias, 4 * l.hidden),
      }));

      this.pinyin = data.pinyin;
      this.freq = data.freq;
      this.bigram = data.bigram;
      this._syllables = Object.keys(this.pinyin).sort((a, b) => b.length - a.length);
      this._syllableSet = new Set(Object.keys(this.pinyin));
      this.loaded = true;
    }

    ids(text) {
      const out = [];
      for (const c of text) {
        const id = this.char2id.get(c);
        if (id !== undefined) out.push(id);
      }
      return out;
    }

    zeroState() {
      const H = this.H;
      return {
        h: Array.from({ length: this.L }, () => new Float32Array(H)),
        c: Array.from({ length: this.L }, () => new Float32Array(H)),
      };
    }

    cloneState(s) {
      return {
        h: s.h.map((a) => Float32Array.from(a)),
        c: s.c.map((a) => Float32Array.from(a)),
      };
    }

    // 单步前向：读入一个字，返回新状态 + 各层门控
    step(id, state) {
      const H = this.H;
      const h = state.h.map((a) => Float32Array.from(a));
      const c = state.c.map((a) => Float32Array.from(a));
      const gatesPerLayer = [];

      let xVec = this.embed.subarray(id * this.E, id * this.E + this.E);
      for (let li = 0; li < this.L; li++) {
        const W = this.layers[li];
        const inDim = W.in;
        const outDim = 4 * H;
        const g = new Float32Array(outDim);
        // W_ih · x + W_hh · h + b   （权重按 (4H, in) 行主序存，逐行点积）
        const wIh = W.wIh;
        for (let j = 0; j < outDim; j++) {
          let acc = W.bias[j];
          let base = j * inDim;
          for (let k = 0; k < inDim; k++) acc += wIh[base + k] * xVec[k];
          base = j * H;
          const wHh = W.wHh;
          for (let k = 0; k < H; k++) acc += wHh[base + k] * h[li][k];
          g[j] = acc;
        }
        const gi = new Float32Array(H), gf = new Float32Array(H), gg = new Float32Array(H), go = new Float32Array(H);
        for (let k = 0; k < H; k++) {
          gi[k] = sigmoid(g[k]);
          gf[k] = sigmoid(g[H + k]);
          gg[k] = tanh(g[2 * H + k]);
          go[k] = sigmoid(g[3 * H + k]);
        }
        for (let k = 0; k < H; k++) {
          c[li][k] = gf[k] * c[li][k] + gi[k] * gg[k];
          h[li][k] = go[k] * tanh(c[li][k]);
        }
        gatesPerLayer.push({ i: gi, f: gf, g: gg, o: go });
        xVec = h[li];
      }
      return { h, c, gates: gatesPerLayer, top: state.h ? null : null };
    }

    // 对最后一层隐状态做 fc，得到整个字表的 logits
    logits(state) {
      const H = this.H, V = this.V;
      const x = state.h[this.L - 1];
      const out = new Float32Array(V);
      for (let v = 0; v < V; v++) {
        let acc = this.fcB[v];
        const base = v * H;
        for (let k = 0; k < H; k++) acc += this.fcW[base + k] * x[k];
        out[v] = acc;
      }
      return out;
    }

    // 把一段文本逐字送进模型，返回每步状态与概率（用于可视化）
    run(text, opts) {
      const temperature = (opts && opts.temperature) || 1;
      const ids = this.ids(text);
      let state = this.zeroState();
      const traces = [];
      for (let t = 0; t < ids.length; t++) {
        const id = ids[t];
        state = this.step(id, state);
        const logits = this.logits(state);
        const logp = logSoftmaxArr(logits, temperature);
        traces.push({
          char: this.vocab[id],
          id,
          layer1: Float32Array.from(state.h[0]),
          layer2: Float32Array.from(state.h[this.L - 1]),
          gates: state.gates,
          logits: logits,
          logp: logp,
        });
      }
      return { ids, state, traces };
    }

    // 上下文 → 下一个字 top-k（真实 softmax 概率）
    nextDist(state, k, temperature) {
      const logits = this.logits(state);
      const logp = logSoftmaxArr(logits, temperature || 1);
      const skip = /\s/;   // 语料一行一句，模型会预测换行；对输入法无用，展示时跳过
      const idx = Array.from(logp.keys())
        .filter((i) => !skip.test(this.vocab[i]))
        .sort((a, b) => logp[b] - logp[a])
        .slice(0, k || 8);
      return {
        total: logp.length,
        entries: idx.map((i) => ({
          id: i,
          char: this.vocab[i],
          logp: logp[i],
          p: Math.exp(logp[i]),
        })),
      };
    }

    // ───────── 拼音切分（与 pinyin_ime.py 的贪心最长匹配一致） ─────────
    segment(text) {
      const syllables = [];
      const tokens = String(text).toLowerCase().trim().split(/\s+/).filter(Boolean);
      for (const token of tokens) {
        let i = 0;
        while (i < token.length) {
          let matched = null;
          for (const s of this._syllables) {
            if (token.startsWith(s, i)) {
              matched = s;
              break;
            }
          }
          if (matched) {
            syllables.push(matched);
            i += matched.length;
          } else {
            i += 1;
          }
        }
      }
      return syllables;
    }

    candidates(syllable) {
      return this.pinyin[syllable] || [];
    }

    // 字频（N-gram unigram 基线）
    unigramLogp() {
      if (this._uniLogp) return this._uniLogp;
      const V = this.V;
      const counts = new Float64Array(V);
      let total = 0;
      for (let i = 0; i < V; i++) {
        const c = this.freq[this.vocab[i]] || 0;
        counts[i] = c + 1; // 平滑
        total += counts[i];
      }
      const out = new Float32Array(V);
      for (let i = 0; i < V; i++) out[i] = Math.log(counts[i] / total);
      this._uniLogp = out;
      return out;
    }

    /**
     * 束搜索解码（输入法核心）
     * 返回每一步的展开/剪枝过程 + 最终候选，供可视化回放。
     */
    beamSearch(syllables, prefix, opts) {
      const o = Object.assign(
        { beam: 10, useLM: true, temperature: 1, topk: 8 },
        opts || {}
      );
      // 先把已确认的上下文跑一遍，得到初始状态 + 逐字轨迹
      const pre = this.run(prefix, { temperature: o.temperature });
      let beams = [{ score: 0, text: "", state: pre.state }];
      const steps = [];

      for (const syllable of syllables) {
        const cands = this.candidates(syllable);
        if (!cands.length) {
          steps.push({ syllable, skipped: true, beams: [] });
          continue;
        }
        const children = [];
        // 每个 beam 只做一次 fc，再对该音节的候选字取分
        for (const b of beams) {
          const ctxLen = prefix.length + b.text.length;
          let logp;
          if (o.useLM && ctxLen > 0) {
            logp = logSoftmaxArr(this.logits(b.state), o.temperature);
          } else {
            // 没有上文（第一个字）→ 退回字频先验，等价于 unigram 语言模型
            logp = this.unigramLogp();
          }
          for (const ch of cands) {
            const id = this.char2id.get(ch);
            if (id === undefined) continue;
            children.push({
              parent: b,
              char: ch,
              id,
              score: b.score + logp[id],
              logp: logp[id],
            });
          }
        }
        children.sort((a, b) => b.score - a.score);
        const kept = children.slice(0, o.beam);
        // 只给存活的路径推进状态
        for (const k of kept) {
          k.state = this.step(k.id, k.parent.state);
          k.text = k.parent.text + k.char;
        }
        steps.push({
          syllable,
          candidates: cands,
          expanded: children.length,
          beamsIn: beams.map((b) => ({ text: b.text, score: b.score })),
          kept: kept.map((k) => ({
            text: k.text,
            char: k.char,
            score: k.score,
            logp: k.logp,
            parent: k.parent.text,
            state: k.state,
          })),
        });
        beams = kept;
      }

      const results = beams
        .map((b) => ({
          text: b.text,
          score: b.score,
          avgLogp: syllables.length ? b.score / syllables.length : 0,
          ppl: Math.exp(-(syllables.length ? b.score / syllables.length : 0)),
        }))
        .sort((a, b) => b.score - a.score);
      return { steps, results, prefixTrace: pre };
    }
  }

  root.IME_ENGINE = { CharLSTM, decodeFp16, logSoftmaxArr, halfToFloat };
})(typeof window !== "undefined" ? window : globalThis);
