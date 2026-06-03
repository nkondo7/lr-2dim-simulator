import React, { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

function pseudoRandom(seed) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

function normalLikeNoise(seed) {
  let sum = 0;
  for (let i = 0; i < 12; i++) sum += pseudoRandom(seed * 100 + i);
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

function formatNumber(value, digits = 2) {
  return Number(value).toFixed(digits);
}

const DATA_PATTERNS = [
  {
    id: "linear_easy",
    label: "A：直線で分けやすい",
    shortLabel: "直線",
    description: "ロジスティック回帰でよく当てはまる基本例",
    fn: (x1, x2) => 1.15 * x1 + 0.9 * x2 - 1.05,
  },
  {
    id: "linear_steep",
    label: "B：傾いた直線",
    shortLabel: "傾き",
    description: "w₁・w₂・bの役割が見えやすい例",
    fn: (x1, x2) => -1.25 * x1 + 1.1 * x2 + 0.12,
  },
  {
    id: "corner",
    label: "C：右上だけ○",
    shortLabel: "右上",
    description: "1本の直線では完全には分けにくい例",
    fn: (x1, x2) => Math.min(x1 - 0.62, x2 - 0.62),
  },
  {
    id: "circle",
    label: "D：中央が○",
    shortLabel: "円形",
    description: "線形モデルの限界を確認する例",
    fn: (x1, x2) => 0.25 - (x1 - 0.5) ** 2 - (x2 - 0.5) ** 2,
  },
  {
    id: "xor",
    label: "E：XOR型",
    shortLabel: "XOR",
    description: "ロジスティック回帰では苦手な典型例",
    fn: (x1, x2) => ((x1 - 0.5) * (x2 - 0.5) < 0 ? 0.22 : -0.22),
  },
];

function makeDefaultParams() {
  return {
    bias: -0.2,
    weight1: 1.2,
    weight2: 1.2,
  };
}

function predictLogit(params, x1, x2) {
  return params.bias + params.weight1 * x1 + params.weight2 * x2;
}

function predictProb(params, x1, x2) {
  return sigmoid(predictLogit(params, x1, x2));
}

function calculateMetrics(data, params) {
  if (data.length === 0) return { loss: 0, accuracy: 0 };
  let loss = 0;
  let correct = 0;
  const eps = 1e-7;

  for (const point of data) {
    const p = clamp(predictProb(params, point.x1, point.x2), eps, 1 - eps);
    loss += -(point.y * Math.log(p) + (1 - point.y) * Math.log(1 - p));
    if ((p >= 0.5 ? 1 : 0) === point.y) correct += 1;
  }

  return { loss: loss / data.length, accuracy: correct / data.length };
}

function gradientStep(data, currentParams, options = {}) {
  const learningRate = options.learningRate ?? 0.08;
  const l2 = options.l2 ?? 0.001;
  const n = Math.max(data.length, 1);

  let gradBias = 0;
  let gradWeight1 = 0;
  let gradWeight2 = 0;

  for (const point of data) {
    const p = predictProb(currentParams, point.x1, point.x2);
    const common = (p - point.y) / n;
    gradBias += common;
    gradWeight1 += common * point.x1;
    gradWeight2 += common * point.x2;
  }

  return {
    bias: clamp(currentParams.bias - learningRate * gradBias, -15, 15),
    weight1: clamp(currentParams.weight1 - learningRate * (gradWeight1 + l2 * currentParams.weight1), -25, 25),
    weight2: clamp(currentParams.weight2 - learningRate * (gradWeight2 + l2 * currentParams.weight2), -25, 25),
  };
}

function trainMultipleSteps(data, currentParams, steps, options) {
  let next = currentParams;
  for (let i = 0; i < steps; i++) next = gradientStep(data, next, options);
  return next;
}

function RangeSlider({ label, value, min, max, step, onChange, disabled = false }) {
  const digits = step < 0.01 ? 3 : step < 0.1 ? 2 : step < 1 ? 1 : 0;
  return (
    <div className={`sliderBlock ${disabled ? "disabled" : ""}`}>
      <div className="sliderHeader">
        <label>{label}</label>
        <span>{formatNumber(value, digits)}</span>
      </div>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </div>
  );
}

function ParameterMiniBar({ value, min = -25, max = 25 }) {
  const zero = ((0 - min) / (max - min)) * 100;
  const pos = ((value - min) / (max - min)) * 100;
  const left = Math.min(zero, pos);
  const width = Math.abs(pos - zero);
  return (
    <div className="miniBar">
      <div className="miniBarZero" style={{ left: `${zero}%` }} />
      <div className="miniBarValue" style={{ left: `${left}%`, width: `${width}%` }} />
    </div>
  );
}

function Button({ children, onClick, variant = "primary", disabled = false, className = "" }) {
  return (
    <button type="button" className={`btn ${variant} ${className}`} onClick={onClick} disabled={disabled}>
      {children}
    </button>
  );
}

function SwitchRow({ label, checked, onChange }) {
  return (
    <label className="switchRow">
      <span>{label}</span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
    </label>
  );
}

export default function App() {
  const [params, setParams] = useState(makeDefaultParams());
  const [sampleSize, setSampleSize] = useState(120);
  const [labelNoise, setLabelNoise] = useState(0.06);
  const [dataSeed, setDataSeed] = useState(1);
  const [inputRangeMode, setInputRangeMode] = useState("unit");
  const [datasetId, setDatasetId] = useState("linear_easy");
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showTrueBoundary, setShowTrueBoundary] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingSteps, setTrainingSteps] = useState(0);
  const [bestLoss, setBestLoss] = useState(null);
  const [learningRate, setLearningRate] = useState(0.08);
  const [stepsPerFrame, setStepsPerFrame] = useState(25);
  const trainingStartedRef = useRef(false);

  const width = 640;
  const height = 520;
  const padding = 46;
  const plotSize = Math.min(width - padding * 2, height - padding * 2);
  const plotLeft = padding;
  const plotTop = padding;
  const selectedPattern = DATA_PATTERNS.find((pattern) => pattern.id === datasetId) ?? DATA_PATTERNS[0];

  const xMin = inputRangeMode === "wide" ? -2 : 0;
  const xMax = inputRangeMode === "wide" ? 3 : 1;
  const yMin = inputRangeMode === "wide" ? -2 : 0;
  const yMax = inputRangeMode === "wide" ? 3 : 1;

  const patternMargin = (x1, x2) => selectedPattern.fn(x1, x2);

  const xToSvg = (x) => plotLeft + ((x - xMin) / (xMax - xMin)) * plotSize;
  const yToSvg = (y) => plotTop + plotSize - ((y - yMin) / (yMax - yMin)) * plotSize;

  const data = useMemo(() => {
    const result = [];
    for (let i = 0; i < sampleSize; i++) {
      const x1 = pseudoRandom(dataSeed * 10000 + i * 11 + 1);
      const x2 = pseudoRandom(dataSeed * 10000 + i * 11 + 2);
      const margin = patternMargin(x1, x2);
      let y = margin >= 0 ? 1 : 0;
      if (pseudoRandom(dataSeed * 10000 + i * 11 + 3) < labelNoise) y = 1 - y;
      const jitterX = clamp(x1 + 0.003 * normalLikeNoise(dataSeed * 10000 + i * 11 + 4), 0, 1);
      const jitterY = clamp(x2 + 0.003 * normalLikeNoise(dataSeed * 10000 + i * 11 + 5), 0, 1);
      result.push({ x1: jitterX, x2: jitterY, y, trueY: margin >= 0 ? 1 : 0 });
    }
    return result;
  }, [sampleSize, labelNoise, dataSeed, selectedPattern]);

  useEffect(() => {
    if (!isTraining) return undefined;
    let animationId;
    const animate = () => {
      setParams((prev) => {
        trainingStartedRef.current = true;
        return trainMultipleSteps(data, prev, stepsPerFrame, { learningRate });
      });
      setTrainingSteps((prev) => prev + stepsPerFrame);
      animationId = requestAnimationFrame(animate);
    };
    animationId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationId);
  }, [isTraining, data, learningRate, stepsPerFrame]);

  const metrics = useMemo(() => calculateMetrics(data, params), [data, params]);

  useEffect(() => {
    setBestLoss((prev) => (prev === null ? metrics.loss : Math.min(prev, metrics.loss)));
  }, [metrics.loss]);

  const boundaryCells = useMemo(() => {
    const cells = [];
    const grid = 54;
    const cellSize = plotSize / grid;
    for (let gy = 0; gy < grid; gy++) {
      for (let gx = 0; gx < grid; gx++) {
        const x1 = xMin + ((gx + 0.5) / grid) * (xMax - xMin);
        const x2 = yMax - ((gy + 0.5) / grid) * (yMax - yMin);
        const p = predictProb(params, x1, x2);
        const opacity = 0.08 + Math.abs(p - 0.5) * 0.5;
        cells.push({ gx, gy, p, opacity, cellSize });
      }
    }
    return cells;
  }, [params, plotSize, inputRangeMode]);

  const contourSegments = useMemo(() => {
    const segments = [];
    const grid = 58;
    const values = Array.from({ length: grid + 1 }, (_, iy) =>
      Array.from({ length: grid + 1 }, (_, ix) => {
        const x1 = xMin + (ix / grid) * (xMax - xMin);
        const x2 = yMin + (iy / grid) * (yMax - yMin);
        return predictProb(params, x1, x2) - 0.5;
      })
    );

    function interp(ix1, iy1, v1, ix2, iy2, v2) {
      const t = Math.abs(v1 - v2) < 1e-9 ? 0.5 : Math.abs(v1) / (Math.abs(v1) + Math.abs(v2));
      const x = xMin + ((ix1 + t * (ix2 - ix1)) / grid) * (xMax - xMin);
      const y = yMin + ((iy1 + t * (iy2 - iy1)) / grid) * (yMax - yMin);
      return [xToSvg(x), yToSvg(y)];
    }

    for (let iy = 0; iy < grid; iy++) {
      for (let ix = 0; ix < grid; ix++) {
        const v00 = values[iy][ix];
        const v10 = values[iy][ix + 1];
        const v11 = values[iy + 1][ix + 1];
        const v01 = values[iy + 1][ix];
        const points = [];
        if (v00 * v10 < 0) points.push(interp(ix, iy, v00, ix + 1, iy, v10));
        if (v10 * v11 < 0) points.push(interp(ix + 1, iy, v10, ix + 1, iy + 1, v11));
        if (v11 * v01 < 0) points.push(interp(ix + 1, iy + 1, v11, ix, iy + 1, v01));
        if (v01 * v00 < 0) points.push(interp(ix, iy + 1, v01, ix, iy, v00));
        if (points.length === 2) segments.push(points);
        if (points.length === 4) {
          segments.push([points[0], points[1]]);
          segments.push([points[2], points[3]]);
        }
      }
    }
    return segments;
  }, [params, inputRangeMode]);

  const trueBoundarySegments = useMemo(() => {
    const segments = [];
    const grid = 58;
    const values = Array.from({ length: grid + 1 }, (_, iy) =>
      Array.from({ length: grid + 1 }, (_, ix) => selectedPattern.fn(ix / grid, iy / grid))
    );

    function interp(ix1, iy1, v1, ix2, iy2, v2) {
      const t = Math.abs(v1 - v2) < 1e-9 ? 0.5 : Math.abs(v1) / (Math.abs(v1) + Math.abs(v2));
      return [xToSvg((ix1 + t * (ix2 - ix1)) / grid), yToSvg((iy1 + t * (iy2 - iy1)) / grid)];
    }

    for (let iy = 0; iy < grid; iy++) {
      for (let ix = 0; ix < grid; ix++) {
        const v00 = values[iy][ix];
        const v10 = values[iy][ix + 1];
        const v11 = values[iy + 1][ix + 1];
        const v01 = values[iy + 1][ix];
        const points = [];
        if (v00 * v10 < 0) points.push(interp(ix, iy, v00, ix + 1, iy, v10));
        if (v10 * v11 < 0) points.push(interp(ix + 1, iy, v10, ix + 1, iy + 1, v11));
        if (v11 * v01 < 0) points.push(interp(ix + 1, iy + 1, v11, ix, iy + 1, v01));
        if (v01 * v00 < 0) points.push(interp(ix, iy + 1, v01, ix, iy, v00));
        if (points.length === 2) segments.push(points);
        if (points.length === 4) {
          segments.push([points[0], points[1]]);
          segments.push([points[2], points[3]]);
        }
      }
    }
    return segments;
  }, [selectedPattern, inputRangeMode]);

  const stopTraining = () => setIsTraining(false);
  const clearTrainingProgress = () => {
    setTrainingSteps(0);
    setBestLoss(null);
    trainingStartedRef.current = false;
  };

  const updateParam = (key, value) => {
    stopTraining();
    clearTrainingProgress();
    setParams((prev) => ({ ...prev, [key]: value }));
  };

  const reset = () => {
    stopTraining();
    setParams(makeDefaultParams());
    setSampleSize(120);
    setLabelNoise(0.06);
    setDataSeed(1);
    setDatasetId("linear_easy");
    setInputRangeMode("unit");
    setShowHeatmap(true);
    setShowTrueBoundary(false);
    setLearningRate(0.08);
    setStepsPerFrame(25);
    clearTrainingProgress();
  };

  const regenerateData = () => {
    stopTraining();
    setDataSeed((prev) => prev + 1);
    clearTrainingProgress();
  };

  const randomizeParams = () => {
    stopTraining();
    setParams({
      bias: (pseudoRandom(dataSeed * 77 + 1) - 0.5) * 6,
      weight1: (pseudoRandom(dataSeed * 77 + 2) - 0.5) * 12,
      weight2: (pseudoRandom(dataSeed * 77 + 3) - 0.5) * 12,
    });
    clearTrainingProgress();
  };

  const handleDatasetChange = (id) => {
    stopTraining();
    setDatasetId(id);
    clearTrainingProgress();
  };

  const axisTicks = inputRangeMode === "wide" ? [-2, -1, 0, 1, 2, 3] : [0, 0.2, 0.4, 0.6, 0.8, 1];
  const gridLines = [];
  for (const v of axisTicks) {
    gridLines.push(<line key={`gx-${v}`} x1={xToSvg(v)} y1={yToSvg(yMin)} x2={xToSvg(v)} y2={yToSvg(yMax)} className="gridLine" />);
    gridLines.push(<line key={`gy-${v}`} x1={xToSvg(xMin)} y1={yToSvg(v)} x2={xToSvg(xMax)} y2={yToSvg(v)} className="gridLine" />);
  }

  const boundaryEquation = Math.abs(params.weight2) > 1e-8
    ? `x₂ = ${formatNumber(-params.weight1 / params.weight2, 2)} x₁ ${-params.bias / params.weight2 >= 0 ? "+" : "-"} ${formatNumber(Math.abs(-params.bias / params.weight2), 2)}`
    : "w₂が0に近いため縦方向の境界";

  return (
    <div className="appShell">
      <div className="mainWrap">
        <header className="topHeader">
          <div>
            <h1>ロジスティック回帰 二値分類シミュレータ</h1>
            <p>線形パラメータ b, w₁, w₂ を動かして、確率と決定境界がどう変わるかを確認できます。</p>
          </div>
          <div className="legend">
            <span><i className="dot blue" />○</span>
            <span><i className="cross red" />×</span>
            <span><i className="lineBlack" />p=0.5</span>
            <span><i className={isTraining ? "dot pulse" : "dot gray"} />探索中: {isTraining ? "ON" : "OFF"}</span>
          </div>
        </header>

        <div className="layoutGrid">
          <div className="leftColumn">
            <section className="card plotCard">
              <div className="plotGrid">
                <svg viewBox={`0 0 ${width} ${height}`} className="plotSvg">
                  {showHeatmap && boundaryCells.map((cell, i) => (
                    <rect
                      key={`cell-${i}`}
                      x={plotLeft + cell.gx * cell.cellSize}
                      y={plotTop + cell.gy * cell.cellSize}
                      width={cell.cellSize + 0.5}
                      height={cell.cellSize + 0.5}
                      fill={cell.p >= 0.5 ? "#4f46e5" : "#e11d48"}
                      opacity={cell.opacity}
                    />
                  ))}

                  {gridLines}
                  <rect x={plotLeft} y={plotTop} width={plotSize} height={plotSize} fill="none" className="plotFrame" />

                  {axisTicks.map((v) => (
                    <React.Fragment key={`label-${v}`}>
                      <text x={xToSvg(v)} y={yToSvg(yMin) + 20} textAnchor="middle" className="axisLabel">{inputRangeMode === "wide" ? v : v.toFixed(1)}</text>
                      <text x={xToSvg(xMin) - 12} y={yToSvg(v) + 4} textAnchor="end" className="axisLabel">{inputRangeMode === "wide" ? v : v.toFixed(1)}</text>
                    </React.Fragment>
                  ))}
                  <text x={xToSvg(xMax) + 24} y={yToSvg(yMin) + 5} className="axisTitle">x₁</text>
                  <text x={xToSvg(xMin) - 8} y={yToSvg(yMax) - 18} className="axisTitle">x₂</text>

                  {showTrueBoundary && trueBoundarySegments.map((seg, i) => (
                    <line key={`true-seg-${i}`} x1={seg[0][0]} y1={seg[0][1]} x2={seg[1][0]} y2={seg[1][1]} className="trueBoundary" />
                  ))}

                  {contourSegments.map((seg, i) => (
                    <line key={`seg-${i}`} x1={seg[0][0]} y1={seg[0][1]} x2={seg[1][0]} y2={seg[1][1]} className="decisionBoundary" />
                  ))}

                  {data.map((point, index) => (
                    point.y === 1 ? (
                      <circle key={`data-${index}`} cx={xToSvg(point.x1)} cy={yToSvg(point.x2)} r="5" className="pointPositive" />
                    ) : (
                      <g key={`data-${index}`} opacity="0.9">
                        <line x1={xToSvg(point.x1) - 4.5} y1={yToSvg(point.x2) - 4.5} x2={xToSvg(point.x1) + 4.5} y2={yToSvg(point.x2) + 4.5} className="pointNegative" />
                        <line x1={xToSvg(point.x1) - 4.5} y1={yToSvg(point.x2) + 4.5} x2={xToSvg(point.x1) + 4.5} y2={yToSvg(point.x2) - 4.5} className="pointNegative" />
                      </g>
                    )
                  ))}
                </svg>

                <aside className="metricsPanel">
                  <div className="metricBox blueBox">
                    <div>正解率</div>
                    <strong>{(metrics.accuracy * 100).toFixed(1)}%</strong>
                  </div>
                  <div className="metricBox redBox">
                    <div>交差エントロピー</div>
                    <strong>{metrics.loss.toFixed(3)}</strong>
                  </div>
                  <div className="metricBox grayBox">
                    <div>最良Loss / 探索回数</div>
                    <strong>{bestLoss === null ? "—" : bestLoss.toFixed(3)} / {trainingSteps}</strong>
                  </div>
                  <div className="formulaBox">
                    <div>p(○)=σ(b+w₁x₁+w₂x₂)</div>
                    <div>p=0.5 ⇔ b+w₁x₁+w₂x₂=0</div>
                    <div>{boundaryEquation}</div>
                  </div>
                  <div className="switchGroup">
                    <SwitchRow label="確率の色分け" checked={showHeatmap} onChange={setShowHeatmap} />
                    <SwitchRow label="真の境界" checked={showTrueBoundary} onChange={setShowTrueBoundary} />
                  </div>
                </aside>
              </div>
            </section>

            <section className="card dataPanel">
              <div className="datasetBlock">
                <div className="smallTitle">○×の分布</div>
                <div className="buttonGrid five">
                  {DATA_PATTERNS.map((pattern) => (
                    <Button
                      key={pattern.id}
                      variant={datasetId === pattern.id ? "primary" : "outline"}
                      disabled={isTraining}
                      onClick={() => handleDatasetChange(pattern.id)}
                    >
                      {pattern.shortLabel}
                    </Button>
                  ))}
                </div>
                <p className="descriptionText">{selectedPattern.label}：{selectedPattern.description}</p>
              </div>
              <div className="rangeBlock">
                <div className="smallTitle">x₁・x₂の表示範囲</div>
                <div className="buttonGrid two">
                  <Button variant={inputRangeMode === "unit" ? "primary" : "outline"} onClick={() => setInputRangeMode("unit")}>0〜1</Button>
                  <Button variant={inputRangeMode === "wide" ? "primary" : "outline"} onClick={() => setInputRangeMode("wide")}>-2〜3</Button>
                </div>
              </div>
              <RangeSlider label="サンプルサイズ" value={sampleSize} min={40} max={300} step={10} disabled={isTraining} onChange={(v) => { stopTraining(); setSampleSize(v); clearTrainingProgress(); }} />
              <RangeSlider label="ラベル反転ノイズ" value={labelNoise} min={0} max={0.25} step={0.01} disabled={isTraining} onChange={(v) => { stopTraining(); setLabelNoise(v); clearTrainingProgress(); }} />
              <Button onClick={regenerateData} variant="outline" disabled={isTraining}>データ再生成</Button>
            </section>
          </div>

          <div className="rightColumn">
            <section className="card trainPanel">
              <div className="buttonGrid two">
                <Button onClick={() => setIsTraining((prev) => !prev)} variant={isTraining ? "danger" : "primary"}>
                  {isTraining ? "探索ストップ" : trainingSteps > 0 ? "探索再開" : "探索スタート"}
                </Button>
                <Button onClick={randomizeParams} variant="outline" disabled={isTraining}>ランダム初期値</Button>
              </div>
              <div className="sliderTwoCols">
                <RangeSlider label="学習率" value={learningRate} min={0.002} max={0.3} step={0.002} onChange={setLearningRate} />
                <RangeSlider label="探索速度" value={stepsPerFrame} min={1} max={120} step={1} onChange={setStepsPerFrame} />
              </div>
            </section>

            <section className="card parameterSummary">
              <div className="summaryHeader">
                <div>
                  <h2>線形パラメータ</h2>
                  <p>パラメータ数：3</p>
                </div>
              </div>
              <div className="paramCards">
                <div className="paramCard">
                  <span>b</span>
                  <strong>{formatNumber(params.bias, 2)}</strong>
                  <ParameterMiniBar value={params.bias} min={-15} max={15} />
                </div>
                <div className="paramCard">
                  <span>w₁</span>
                  <strong>{formatNumber(params.weight1, 2)}</strong>
                  <ParameterMiniBar value={params.weight1} />
                </div>
                <div className="paramCard">
                  <span>w₂</span>
                  <strong>{formatNumber(params.weight2, 2)}</strong>
                  <ParameterMiniBar value={params.weight2} />
                </div>
              </div>
            </section>

            <section className="card slidersPanel">
              <h2>手動で編集</h2>
              <div className="sliderStack">
                <RangeSlider label="b：切片 / バイアス" value={params.bias} min={-15} max={15} step={0.05} disabled={isTraining} onChange={(value) => updateParam("bias", value)} />
                <RangeSlider label="w₁：x₁の重み" value={params.weight1} min={-25} max={25} step={0.05} disabled={isTraining} onChange={(value) => updateParam("weight1", value)} />
                <RangeSlider label="w₂：x₂の重み" value={params.weight2} min={-25} max={25} step={0.05} disabled={isTraining} onChange={(value) => updateParam("weight2", value)} />
              </div>
            </section>

            <section className="card memoPanel">
              <h2>見るポイント</h2>
              <ul>
                <li>w₁やw₂の符号を変えると、○になりやすい方向が変わります。</li>
                <li>bを動かすと、決定境界が平行移動します。</li>
                <li>重みの絶対値を大きくすると、境界付近の確率変化が急になります。</li>
                <li>円形やXOR型では、線形モデルだけでは限界があることが見えます。</li>
              </ul>
            </section>

            <div className="footerButtons">
              <Button onClick={reset} variant="outline">初期値</Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
