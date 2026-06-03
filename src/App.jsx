import React, { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

function pseudoRandom(seed) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

function normalLikeNoise(seed) {
  let sum = 0;
  for (let i = 0; i < 12; i += 1) sum += pseudoRandom(seed * 100 + i);
  return sum - 6;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function sigmoid(z) {
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

function fmt(value, digits = 2) {
  return Number(value).toFixed(digits);
}

const DATA_PATTERNS = [
  {
    id: "linear",
    label: "直線",
    desc: "x₁+x₂が大きいほど○",
    fn: (x1, x2) => x1 + x2 - 1.03,
    init: { b: -6, w1: 6, w2: 6 },
  },
  {
    id: "vertical",
    label: "縦境界",
    desc: "x₁だけで分かれる",
    fn: (x1) => x1 - 0.52,
    init: { b: -4, w1: 8, w2: 0 },
  },
  {
    id: "horizontal",
    label: "横境界",
    desc: "x₂だけで分かれる",
    fn: (_x1, x2) => x2 - 0.52,
    init: { b: -4, w1: 0, w2: 8 },
  },
  {
    id: "corner",
    label: "右上",
    desc: "線形では少し苦手",
    fn: (x1, x2) => Math.min(x1 - 0.6, x2 - 0.6),
    init: { b: -7, w1: 5, w2: 5 },
  },
  {
    id: "circle",
    label: "円形",
    desc: "線形モデルの限界",
    fn: (x1, x2) => 0.24 - (x1 - 0.5) ** 2 - (x2 - 0.5) ** 2,
    init: { b: 0, w1: 0, w2: 0 },
  },
  {
    id: "xor",
    label: "XOR",
    desc: "線形モデルの限界",
    fn: (x1, x2) => ((x1 - 0.5) * (x2 - 0.5) < 0 ? 0.22 : -0.22),
    init: { b: 0, w1: 0, w2: 0 },
  },
];

const DEFAULT_STATE = {
  b: 0,
  w1: 3,
  w2: 3,
  sampleSize: 120,
  labelNoise: 0.06,
  seed: 1,
  dataset: "linear",
  showHeatmap: true,
  showTrue: true,
  training: false,
  steps: 0,
  bestLoss: null,
  lr: 0.08,
  spf: 25,
  rotZ: -38,
  rotX: 58,
  quality: "fast",
  rangeMode: "unit",
};

function RangeControl({ label, value, min, max, step, digits = 2, onChange }) {
  return (
    <div className="control">
      <div className="controlTop">
        <label>{label}</label>
        <output>{fmt(value, digits)}</output>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        onInput={(event) => onChange(Number(event.currentTarget.value))}
      />
    </div>
  );
}

function getDomain(rangeMode) {
  if (rangeMode === "wide") return { min: -2, max: 3, ticks: [-2, -1, 0, 1, 2, 3], label: "-2〜3" };
  return { min: 0, max: 1, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1], label: "0〜1" };
}

function tickLabel(v) {
  return Number.isInteger(v) ? String(v) : v.toFixed(1);
}

export default function App() {
  const [model, setModel] = useState(DEFAULT_STATE);
  const modelRef = useRef(model);
  const stopRef = useRef(false);
  const rafRef = useRef(null);
  const timeoutRef = useRef(null);
  const drag3dRef = useRef(null);

  useEffect(() => {
    modelRef.current = model;
  }, [model]);

  const pattern = useMemo(
    () => DATA_PATTERNS.find((item) => item.id === model.dataset) ?? DATA_PATTERNS[0],
    [model.dataset]
  );

  const data = useMemo(() => {
    const points = [];
    for (let i = 0; i < model.sampleSize; i += 1) {
      const x1 = pseudoRandom(model.seed * 10000 + i * 11 + 1);
      const x2 = pseudoRandom(model.seed * 10000 + i * 11 + 2);
      const margin = pattern.fn(x1, x2);
      let y = margin >= 0 ? 1 : 0;
      if (pseudoRandom(model.seed * 10000 + i * 11 + 3) < model.labelNoise) y = 1 - y;
      points.push({
        x1: clamp(x1 + 0.003 * normalLikeNoise(model.seed * 10000 + i * 11 + 4), 0, 1),
        x2: clamp(x2 + 0.003 * normalLikeNoise(model.seed * 10000 + i * 11 + 5), 0, 1),
        y,
        trueY: margin >= 0 ? 1 : 0,
      });
    }
    return points;
  }, [model.seed, model.sampleSize, model.labelNoise, pattern]);

  const dataRef = useRef(data);
  useEffect(() => {
    dataRef.current = data;
  }, [data]);

  const domain = getDomain(model.rangeMode);
  const predictLogit = (x1, x2, source = model) => source.b + source.w1 * x1 + source.w2 * x2;
  const predict = (x1, x2, source = model) => sigmoid(predictLogit(x1, x2, source));
  const norm = (value) => (value - domain.min) / (domain.max - domain.min);
  const denorm = (t) => domain.min + t * (domain.max - domain.min);

  const metrics = useMemo(() => {
    let loss = 0;
    let correct = 0;
    const eps = 1e-7;
    for (const point of data) {
      const p = clamp(predict(point.x1, point.x2), eps, 1 - eps);
      loss += -(point.y * Math.log(p) + (1 - point.y) * Math.log(1 - p));
      if ((p >= 0.5 ? 1 : 0) === point.y) correct += 1;
    }
    return { loss: loss / data.length, accuracy: correct / data.length };
  }, [data, model.b, model.w1, model.w2]);

  useEffect(() => {
    setModel((prev) => ({
      ...prev,
      bestLoss: prev.bestLoss === null ? metrics.loss : Math.min(prev.bestLoss, metrics.loss),
    }));
  }, [metrics.loss]);

  const stopTraining = () => {
    stopRef.current = true;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    rafRef.current = null;
    timeoutRef.current = null;
    setModel((prev) => ({ ...prev, training: false }));
  };

  const resetProgress = () => {
    setModel((prev) => ({ ...prev, steps: 0, bestLoss: null }));
  };

  const gradientStep = (source) => {
    let gb = 0;
    let gw1 = 0;
    let gw2 = 0;
    const currentData = dataRef.current;
    const n = currentData.length;
    for (const point of currentData) {
      const common = (predict(point.x1, point.x2, source) - point.y) / n;
      gb += common;
      gw1 += common * point.x1;
      gw2 += common * point.x2;
    }
    const l2 = 0.0005;
    return {
      ...source,
      b: clamp(source.b - source.lr * gb, -16, 16),
      w1: clamp(source.w1 - source.lr * (gw1 + l2 * source.w1), -24, 24),
      w2: clamp(source.w2 - source.lr * (gw2 + l2 * source.w2), -24, 24),
    };
  };

  const trainLoop = () => {
    if (stopRef.current) return;
    const totalSteps = Math.max(1, Math.floor(modelRef.current.spf));
    const chunkSize = Math.min(10, totalSteps);
    let done = 0;

    const runChunk = () => {
      if (stopRef.current) return;
      let next = modelRef.current;
      const end = Math.min(totalSteps, done + chunkSize);
      for (; done < end; done += 1) next = gradientStep(next);
      modelRef.current = next;

      if (done < totalSteps) {
        timeoutRef.current = setTimeout(runChunk, 0);
        return;
      }

      setModel((prev) => {
        let updated = { ...next, training: true, steps: prev.steps + totalSteps };
        const m = calculateMetrics(dataRef.current, updated);
        updated = {
          ...updated,
          bestLoss: prev.bestLoss === null ? m.loss : Math.min(prev.bestLoss, m.loss),
        };
        modelRef.current = updated;
        return updated;
      });
      rafRef.current = requestAnimationFrame(trainLoop);
    };

    runChunk();
  };

  const toggleTraining = () => {
    if (modelRef.current.training) {
      stopTraining();
      return;
    }
    stopRef.current = false;
    setModel((prev) => {
      const next = { ...prev, training: true };
      modelRef.current = next;
      return next;
    });
    rafRef.current = requestAnimationFrame(trainLoop);
  };

  useEffect(() => () => stopTraining(), []);

  const updateParam = (key, value, options = {}) => {
    if (options.stop !== false) stopTraining();
    setModel((prev) => ({ ...prev, [key]: value, steps: 0, bestLoss: null }));
  };

  const choosePattern = (id) => {
    stopTraining();
    const nextPattern = DATA_PATTERNS.find((item) => item.id === id) ?? DATA_PATTERNS[0];
    setModel((prev) => ({
      ...prev,
      dataset: id,
      b: nextPattern.init.b,
      w1: nextPattern.init.w1,
      w2: nextPattern.init.w2,
      steps: 0,
      bestLoss: null,
    }));
  };

  const randomize = () => {
    stopTraining();
    setModel((prev) => ({
      ...prev,
      b: (pseudoRandom(prev.seed * 77 + 1) - 0.5) * 6,
      w1: (pseudoRandom(prev.seed * 77 + 2) - 0.5) * 14,
      w2: (pseudoRandom(prev.seed * 77 + 3) - 0.5) * 14,
      steps: 0,
      bestLoss: null,
    }));
  };

  const regenerateData = () => {
    stopTraining();
    setModel((prev) => ({ ...prev, seed: prev.seed + 1, steps: 0, bestLoss: null }));
  };

  const applyRecommended = () => {
    stopTraining();
    setModel((prev) => ({ ...prev, ...pattern.init, steps: 0, bestLoss: null }));
  };

  const resetAll = () => {
    stopTraining();
    setModel(DEFAULT_STATE);
  };

  const boundarySegment = () => {
    const points = [];
    const min = domain.min;
    const max = domain.max;
    if (Math.abs(model.w2) > 1e-9) {
      for (const x of [min, max]) {
        const y = -(model.b + model.w1 * x) / model.w2;
        if (y >= min && y <= max) points.push([x, y]);
      }
    }
    if (Math.abs(model.w1) > 1e-9) {
      for (const y of [min, max]) {
        const x = -(model.b + model.w2 * y) / model.w1;
        if (x >= min && x <= max) points.push([x, y]);
      }
    }
    const unique = [];
    for (const point of points) {
      if (!unique.some((item) => Math.hypot(item[0] - point[0], item[1] - point[1]) < 1e-6)) unique.push(point);
    }
    return unique.length >= 2 ? [unique[0], unique[1]] : null;
  };

  const trueBoundarySegments = useMemo(() => {
    const segments = [];
    if (!model.showTrue) return segments;
    const grid = model.quality === "fast" ? 34 : 56;
    const values = Array.from({ length: grid + 1 }, (_, iy) =>
      Array.from({ length: grid + 1 }, (_, ix) => {
        const x = denorm(ix / grid);
        const y = denorm(iy / grid);
        return pattern.fn(x, y);
      })
    );
    const interp = (ix1, iy1, v1, ix2, iy2, v2) => {
      const t = Math.abs(v1 - v2) < 1e-9 ? 0.5 : Math.abs(v1) / (Math.abs(v1) + Math.abs(v2));
      return [denorm((ix1 + t * (ix2 - ix1)) / grid), denorm((iy1 + t * (iy2 - iy1)) / grid)];
    };
    for (let iy = 0; iy < grid; iy += 1) {
      for (let ix = 0; ix < grid; ix += 1) {
        const v00 = values[iy][ix];
        const v10 = values[iy][ix + 1];
        const v11 = values[iy + 1][ix + 1];
        const v01 = values[iy + 1][ix];
        const ps = [];
        if (v00 * v10 < 0) ps.push(interp(ix, iy, v00, ix + 1, iy, v10));
        if (v10 * v11 < 0) ps.push(interp(ix + 1, iy, v10, ix + 1, iy + 1, v11));
        if (v11 * v01 < 0) ps.push(interp(ix + 1, iy + 1, v11, ix, iy + 1, v01));
        if (v01 * v00 < 0) ps.push(interp(ix, iy + 1, v01, ix, iy, v00));
        if (ps.length === 2) segments.push(ps);
        if (ps.length === 4) segments.push([ps[0], ps[1]], [ps[2], ps[3]]);
      }
    }
    return segments;
  }, [model.showTrue, model.quality, model.rangeMode, pattern, model.b, model.w1, model.w2]);

  const draw2d = () => {
    const width = 520;
    const height = 520;
    const pad = 46;
    const size = width - pad * 2;
    const xToSvg = (x) => pad + norm(x) * size;
    const yToSvg = (y) => pad + size - norm(y) * size;
    const gridCount = model.quality === "fast" ? 32 : 52;
    const cells = [];
    for (let gy = 0; gy < gridCount; gy += 1) {
      for (let gx = 0; gx < gridCount; gx += 1) {
        const x = denorm((gx + 0.5) / gridCount);
        const y = denorm(1 - (gy + 0.5) / gridCount);
        const p = predict(x, y);
        cells.push({ gx, gy, p });
      }
    }
    const boundary = boundarySegment();

    return (
      <svg viewBox={`0 0 ${width} ${height}`} className="plotSvg plotSvg2d" aria-label="2D plot">
        {model.showHeatmap &&
          cells.map((cell) => (
            <rect
              key={`${cell.gx}-${cell.gy}`}
              x={pad + cell.gx * (size / gridCount)}
              y={pad + cell.gy * (size / gridCount)}
              width={size / gridCount + 0.5}
              height={size / gridCount + 0.5}
              fill={cell.p >= 0.5 ? "#4f46e5" : "#e11d48"}
              opacity={0.08 + Math.abs(cell.p - 0.5) * 0.45}
            />
          ))}
        {domain.ticks.map((tick) => (
          <React.Fragment key={tick}>
            <line x1={xToSvg(tick)} y1={yToSvg(domain.min)} x2={xToSvg(tick)} y2={yToSvg(domain.max)} className="gridLine" />
            <line x1={xToSvg(domain.min)} y1={yToSvg(tick)} x2={xToSvg(domain.max)} y2={yToSvg(tick)} className="gridLine" />
            <text x={xToSvg(tick)} y={yToSvg(domain.min) + 20} textAnchor="middle" className="tickText">
              {tickLabel(tick)}
            </text>
            <text x={xToSvg(domain.min) - 11} y={yToSvg(tick) + 4} textAnchor="end" className="tickText">
              {tickLabel(tick)}
            </text>
          </React.Fragment>
        ))}
        <rect x={pad} y={pad} width={size} height={size} fill="none" className="axis" />
        <text x={xToSvg(domain.max) + 18} y={yToSvg(domain.min) + 5} className="axisLabel">x₁</text>
        <text x={xToSvg(domain.min) - 8} y={yToSvg(domain.max) - 16} className="axisLabel">x₂</text>
        {trueBoundarySegments.map((segment, index) => (
          <line
            key={`true-${index}`}
            x1={xToSvg(segment[0][0])}
            y1={yToSvg(segment[0][1])}
            x2={xToSvg(segment[1][0])}
            y2={yToSvg(segment[1][1])}
            stroke="#059669"
            strokeWidth="2"
            strokeDasharray="6 5"
            opacity="0.75"
          />
        ))}
        {boundary && (
          <line
            x1={xToSvg(boundary[0][0])}
            y1={yToSvg(boundary[0][1])}
            x2={xToSvg(boundary[1][0])}
            y2={yToSvg(boundary[1][1])}
            stroke="#020617"
            strokeWidth="4"
            strokeLinecap="round"
          />
        )}
        {data.map((point, index) => {
          if (point.x1 < domain.min || point.x1 > domain.max || point.x2 < domain.min || point.x2 > domain.max) return null;
          if (point.y === 1) {
            return <circle key={index} cx={xToSvg(point.x1)} cy={yToSvg(point.x2)} r="5" fill="#4f46e5" stroke="white" strokeWidth="1.4" />;
          }
          return (
            <g key={index}>
              <line x1={xToSvg(point.x1) - 4.5} y1={yToSvg(point.x2) - 4.5} x2={xToSvg(point.x1) + 4.5} y2={yToSvg(point.x2) + 4.5} stroke="#e11d48" strokeWidth="2.4" strokeLinecap="round" />
              <line x1={xToSvg(point.x1) - 4.5} y1={yToSvg(point.x2) + 4.5} x2={xToSvg(point.x1) + 4.5} y2={yToSvg(point.x2) - 4.5} stroke="#e11d48" strokeWidth="2.4" strokeLinecap="round" />
            </g>
          );
        })}
      </svg>
    );
  };

  const draw3d = () => {
    const width = 980;
    const height = 720;
    const cx = width * 0.5;
    const cy = height * 0.55;
    const scale = model.rangeMode === "wide" ? 300 : 360;
    const zScale = 270;
    const rz = (model.rotZ * Math.PI) / 180;
    const rx = (model.rotX * Math.PI) / 180;
    const cZ = Math.cos(rz);
    const sZ = Math.sin(rz);
    const cX = Math.cos(rx);
    const sX = Math.sin(rx);
    const project = (x, y, z) => {
      const nx = norm(x) - 0.5;
      const ny = norm(y) - 0.5;
      const nz = z - 0.5;
      const xr = nx * cZ - ny * sZ;
      const yr = nx * sZ + ny * cZ;
      const y2 = yr * cX - nz * sX;
      return [cx + xr * scale, cy + y2 * scale - nz * zScale * 0.2];
    };
    const gridCount = model.quality === "fast" ? 18 : 30;
    const cells = [];
    for (let iy = 0; iy < gridCount; iy += 1) {
      for (let ix = 0; ix < gridCount; ix += 1) {
        const x0 = denorm(ix / gridCount);
        const x1 = denorm((ix + 1) / gridCount);
        const y0 = denorm(iy / gridCount);
        const y1 = denorm((iy + 1) / gridCount);
        const z00 = predict(x0, y0);
        const z10 = predict(x1, y0);
        const z11 = predict(x1, y1);
        const z01 = predict(x0, y1);
        const mid = (z00 + z10 + z11 + z01) / 4;
        cells.push({
          points: [project(x0, y0, z00), project(x1, y0, z10), project(x1, y1, z11), project(x0, y1, z01)],
          depth: ix + iy + mid,
          p: mid,
        });
      }
    }
    cells.sort((a, b) => a.depth - b.depth);
    const boundary = boundarySegment();
    const axes = [
      [[domain.min, domain.min, 0], [domain.max, domain.min, 0], "x₁"],
      [[domain.min, domain.min, 0], [domain.min, domain.max, 0], "x₂"],
      [[domain.min, domain.min, 0], [domain.min, domain.min, 1], "p"],
    ];
    const sourcePoints = model.quality === "fast" && data.length > 150 ? data.filter((_, index) => index % 2 === 0) : data;
    const sortedPoints = [...sourcePoints].sort((a, b) => norm(a.x1) + norm(a.x2) + a.y - (norm(b.x1) + norm(b.x2) + b.y));

    return (
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="plotSvg plotSvg3d"
        aria-label="3D probability surface"
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          drag3dRef.current = { x: event.clientX, y: event.clientY, rotZ: model.rotZ, rotX: model.rotX };
        }}
        onPointerMove={(event) => {
          if (!drag3dRef.current) return;
          const drag = drag3dRef.current;
          setModel((prev) => ({
            ...prev,
            rotZ: clamp(drag.rotZ + (event.clientX - drag.x) * 0.35, -90, 20),
            rotX: clamp(drag.rotX - (event.clientY - drag.y) * 0.28, 25, 80),
          }));
        }}
        onPointerUp={() => {
          drag3dRef.current = null;
        }}
        onPointerCancel={() => {
          drag3dRef.current = null;
        }}
      >
        <rect width={width} height={height} fill="#ffffff" rx="18" />
        {cells.map((cell, index) => (
          <polygon
            key={index}
            points={cell.points.map((point) => point.join(",")).join(" ")}
            fill={cell.p >= 0.5 ? "#4f46e5" : "#e11d48"}
            opacity="0.28"
            stroke={cell.p >= 0.5 ? "#4f46e5" : "#e11d48"}
            strokeWidth="0.35"
          />
        ))}
        {boundary && (
          <>
            <line
              x1={project(boundary[0][0], boundary[0][1], 0.5)[0]}
              y1={project(boundary[0][0], boundary[0][1], 0.5)[1]}
              x2={project(boundary[1][0], boundary[1][1], 0.5)[0]}
              y2={project(boundary[1][0], boundary[1][1], 0.5)[1]}
              stroke="#020617"
              strokeWidth="4"
              strokeLinecap="round"
            />
            <line
              x1={project(boundary[0][0], boundary[0][1], 0)[0]}
              y1={project(boundary[0][0], boundary[0][1], 0)[1]}
              x2={project(boundary[1][0], boundary[1][1], 0)[0]}
              y2={project(boundary[1][0], boundary[1][1], 0)[1]}
              stroke="#020617"
              strokeWidth="2"
              strokeDasharray="4 4"
              opacity="0.45"
            />
          </>
        )}
        {sortedPoints.map((point, index) => {
          if (point.x1 < domain.min || point.x1 > domain.max || point.x2 < domain.min || point.x2 > domain.max) return null;
          const base = project(point.x1, point.x2, 0);
          const top = project(point.x1, point.x2, point.y);
          return (
            <g key={index}>
              <line x1={base[0]} y1={base[1]} x2={top[0]} y2={top[1]} className="obsLine" />
              {point.y === 1 ? (
                <circle cx={top[0]} cy={top[1]} r="4.3" fill="#4f46e5" stroke="white" strokeWidth="1.2" />
              ) : (
                <g>
                  <line x1={top[0] - 4} y1={top[1] - 4} x2={top[0] + 4} y2={top[1] + 4} stroke="#e11d48" strokeWidth="2.2" strokeLinecap="round" />
                  <line x1={top[0] - 4} y1={top[1] + 4} x2={top[0] + 4} y2={top[1] - 4} stroke="#e11d48" strokeWidth="2.2" strokeLinecap="round" />
                </g>
              )}
            </g>
          );
        })}
        {axes.map((axis) => {
          const start = project(...axis[0]);
          const end = project(...axis[1]);
          return (
            <g key={axis[2]}>
              <line x1={start[0]} y1={start[1]} x2={end[0]} y2={end[1]} stroke="#1e293b" strokeWidth="2.2" />
              <text x={end[0] + 6} y={end[1] + 4} className="tickText axisLabel">
                {axis[2]}
              </text>
            </g>
          );
        })}
      </svg>
    );
  };

  const formula = `p(○)=σ(${fmt(model.b)} ${model.w1 >= 0 ? "+" : "-"} ${fmt(Math.abs(model.w1))}x₁ ${model.w2 >= 0 ? "+" : "-"} ${fmt(Math.abs(model.w2))}x₂)`;

  return (
    <div className="app">
      <div className="wrap">
        <header className="header">
          <div className="title">
            <h1>ロジスティック回帰 3Dシミュレータ</h1>
            <p>線形パラメータを手動で動かし、勾配法でも学習できます。</p>
          </div>
          <div className="legend">
            <span><span className="dot blueDot" />○ y=1</span>
            <span><span className="cross roseCross" />× y=0</span>
            <span>黒線: p(○)=0.5</span>
            <span>探索中: {model.training ? "ON" : "OFF"}</span>
          </div>
        </header>

        <main className="gridLayout">
          <section>
            <div className="plots">
              <div className="card">
                <div className="body">
                  <div className="plotTitle">
                    <h2>2D：x₁-x₂平面</h2>
                    <span>色は予測確率</span>
                  </div>
                  {draw2d()}
                </div>
              </div>

              <div className="card">
                <div className="body">
                  <div className="plotTitle">
                    <h2>3D：x₁-x₂-p(○)</h2>
                    <span>点は y=0/1 に配置・ドラッグで回転</span>
                  </div>
                  {draw3d()}
                </div>
              </div>
            </div>

            <div className="card dataCard">
              <div className="body dataControls">
                <div>
                  <div className="smallLabel">○×の分布</div>
                  <div className="buttonRow">
                    {DATA_PATTERNS.map((item) => (
                      <button key={item.id} type="button" className={`btn ${model.dataset === item.id ? "active" : ""}`} onClick={() => choosePattern(item.id)}>
                        {item.label}
                      </button>
                    ))}
                  </div>
                  <div className="note">{pattern.desc}</div>
                </div>

                <div>
                  <div className="smallLabel">x₁・x₂の表示範囲</div>
                  <div className="buttons">
                    <button type="button" className={`btn ${model.rangeMode === "unit" ? "active" : ""}`} onClick={() => updateParam("rangeMode", "unit", { stop: false })}>0〜1</button>
                    <button type="button" className={`btn ${model.rangeMode === "wide" ? "active" : ""}`} onClick={() => updateParam("rangeMode", "wide", { stop: false })}>-2〜3</button>
                  </div>
                  <div className="note">データはそのまま、表示だけ切替</div>
                </div>

                <RangeControl label="サンプルサイズ" value={model.sampleSize} min={40} max={300} step={10} digits={0} onChange={(value) => updateParam("sampleSize", value)} />
                <RangeControl label="ラベル反転ノイズ" value={model.labelNoise} min={0} max={0.25} step={0.01} digits={2} onChange={(value) => updateParam("labelNoise", value)} />
                <button type="button" className="btn" onClick={regenerateData}>データ再生成</button>
              </div>
            </div>
          </section>

          <aside className="side">
            <div className="card">
              <div className="body">
                <div className="metrics">
                  <div className="metric blue">
                    <div className="label">正解率</div>
                    <div className="value">{(metrics.accuracy * 100).toFixed(1)}%</div>
                  </div>
                  <div className="metric rose">
                    <div className="label">交差エントロピー</div>
                    <div className="value">{metrics.loss.toFixed(3)}</div>
                  </div>
                  <div className="metric slate fullMetric">
                    <div className="label">最良Loss / 探索回数</div>
                    <div className="value smallValue">{model.bestLoss === null ? "—" : model.bestLoss.toFixed(3)} / {model.steps}</div>
                  </div>
                </div>
                <div className="formula">{formula}</div>
              </div>
            </div>

            <div className="card">
              <div className="body">
                <div className="buttons">
                  <button
                    type="button"
                    className={`btn ${model.training ? "danger" : "primary"}`}
                    onPointerDown={(event) => {
                      if (modelRef.current.training) {
                        event.preventDefault();
                        event.stopPropagation();
                        stopTraining();
                      }
                    }}
                    onClick={toggleTraining}
                  >
                    {model.training ? "探索ストップ" : model.steps > 0 ? "探索再開" : "探索スタート"}
                  </button>
                  <button
                    type="button"
                    className="btn"
                    onPointerDown={(event) => {
                      event.preventDefault();
                      randomize();
                    }}
                    onClick={(event) => event.preventDefault()}
                  >
                    ランダム初期値
                  </button>
                </div>
                <RangeControl label="学習率" value={model.lr} min={0.002} max={0.22} step={0.002} digits={3} onChange={(value) => updateParam("lr", value, { stop: false })} />
                <RangeControl label="探索速度" value={model.spf} min={1} max={120} step={1} digits={0} onChange={(value) => updateParam("spf", value, { stop: false })} />
              </div>
            </div>

            <div className="card">
              <div className="body">
                <div className="smallLabel">線形パラメータ</div>
                <RangeControl label="b：切片" value={model.b} min={-16} max={16} step={0.05} digits={2} onChange={(value) => updateParam("b", value)} />
                <RangeControl label="w₁：x₁の重み" value={model.w1} min={-24} max={24} step={0.05} digits={2} onChange={(value) => updateParam("w1", value)} />
                <RangeControl label="w₂：x₂の重み" value={model.w2} min={-24} max={24} step={0.05} digits={2} onChange={(value) => updateParam("w2", value)} />
                <div className="switches">
                  <label className="switch">確率の色分け<input type="checkbox" checked={model.showHeatmap} onChange={(event) => setModel((prev) => ({ ...prev, showHeatmap: event.target.checked }))} /></label>
                  <label className="switch">真の境界<input type="checkbox" checked={model.showTrue} onChange={(event) => setModel((prev) => ({ ...prev, showTrue: event.target.checked }))} /></label>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="body">
                <div className="smallLabel">3Dの見え方</div>
                <RangeControl label="横回転" value={model.rotZ} min={-90} max={20} step={1} digits={0} onChange={(value) => updateParam("rotZ", value, { stop: false })} />
                <RangeControl label="縦回転" value={model.rotX} min={25} max={80} step={1} digits={0} onChange={(value) => updateParam("rotX", value, { stop: false })} />
                <div className="smallLabel qualityLabel">描画品質</div>
                <div className="buttons">
                  <button type="button" className={`btn ${model.quality === "fast" ? "active" : ""}`} onClick={() => updateParam("quality", "fast", { stop: false })}>軽量</button>
                  <button type="button" className={`btn ${model.quality === "detail" ? "active" : ""}`} onClick={() => updateParam("quality", "detail", { stop: false })}>詳細</button>
                </div>
                <div className="note">軽量では曲面メッシュと境界計算を粗くして、操作を軽くしています。</div>
              </div>
            </div>

            <div className="footerBtns">
              <button type="button" className="btn" onClick={applyRecommended}>おすすめ初期値</button>
              <button type="button" className="btn" onClick={resetAll}>初期値</button>
            </div>
          </aside>
        </main>
      </div>
    </div>
  );
}

function calculateMetrics(data, model) {
  let loss = 0;
  let correct = 0;
  const eps = 1e-7;
  for (const point of data) {
    const p = clamp(sigmoid(model.b + model.w1 * point.x1 + model.w2 * point.x2), eps, 1 - eps);
    loss += -(point.y * Math.log(p) + (1 - point.y) * Math.log(1 - p));
    if ((p >= 0.5 ? 1 : 0) === point.y) correct += 1;
  }
  return { loss: loss / data.length, accuracy: correct / data.length };
}
