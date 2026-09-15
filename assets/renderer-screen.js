(() => {
  "use strict";

  const DATA_ROOT = "assets/renderer-screen/";
  const DEFAULT_LOSS = "bidirectional_cumulative_energy";
  const DEFAULT_CELL = 11; // Fourier onset placement, Thiran propagation.
  const GRID_POINTS = 25;
  const lossLabels = {
    waveform_l1: "L₁",
    waveform_mse: "L₂",
    smooth_mss: "MSS",
    sot_published_composite: "SOT",
    linear_jtfot: "TFW₂",
    log_jtfot: "TFW₂ (1 s = 1 oct)",
    bidirectional_cumulative_energy: "BiCuL",
    log_quadrature_bicul: "Log-Q BiCuL",
  };
  const canvas = document.querySelector("#loss-landscape");
  const context = canvas.getContext("2d");
  const selector = document.querySelector("#loss-selector");
  const status = document.querySelector("#renderer-status");
  const title = document.querySelector("#landscape-title");
  const renderer = document.querySelector("#landscape-renderer");
  const score = document.querySelector("#landscape-score");
  const caption = document.querySelector("#renderer-caption");
  const cellButtons = [...document.querySelectorAll("[data-cell]")];
  const cache = new Map();
  let manifest;
  let selectedLoss = DEFAULT_LOSS;
  let selectedCell = DEFAULT_CELL;
  let currentData;

  const magmaStops = [
    [0.00, [0, 0, 4]],
    [0.13, [28, 16, 68]],
    [0.25, [79, 18, 123]],
    [0.38, [129, 37, 129]],
    [0.50, [182, 54, 121]],
    [0.63, [229, 80, 100]],
    [0.75, [251, 135, 97]],
    [0.88, [254, 194, 135]],
    [1.00, [252, 253, 191]],
  ];

  function magma(value) {
    const bounded = Math.max(0, Math.min(1, value));
    let upper = magmaStops.findIndex(([position]) => position >= bounded);
    if (upper <= 0) return `rgb(${magmaStops[0][1].join(",")})`;
    if (upper < 0) upper = magmaStops.length - 1;
    const [lowPosition, low] = magmaStops[upper - 1];
    const [highPosition, high] = magmaStops[upper];
    const fraction = (bounded - lowPosition) / (highPosition - lowPosition);
    const rgb = low.map((channel, index) => Math.round(channel + fraction * (high[index] - channel)));
    return `rgb(${rgb.join(",")})`;
  }

  async function fetchJson(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    return response.json();
  }

  function setStatus(message, isError = false) {
    status.textContent = message;
    status.classList.toggle("error", isError);
  }

  function buildLossSelector() {
    selector.replaceChildren();
    manifest.losses.forEach((loss) => {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.loss = loss.id;
      button.setAttribute("role", "radio");
      button.setAttribute("aria-checked", String(loss.id === selectedLoss));
      button.textContent = lossLabels[loss.id] || loss.label;
      button.addEventListener("click", () => selectLoss(loss.id));
      selector.append(button);
    });
  }

  async function loadLoss(lossId) {
    if (cache.has(lossId)) return cache.get(lossId);
    const entry = manifest.losses.find((loss) => loss.id === lossId);
    if (!entry) throw new Error(`Unknown loss ${lossId}`);
    const promise = fetchJson(`${DATA_ROOT}${entry.path}`);
    cache.set(lossId, promise);
    return promise;
  }

  async function selectLoss(lossId) {
    selectedLoss = lossId;
    [...selector.querySelectorAll("button")].forEach((button) => {
      button.setAttribute("aria-checked", String(button.dataset.loss === lossId));
    });
    cellButtons.forEach((button) => { button.disabled = true; });
    const label = lossLabels[lossId] || manifest.losses.find((loss) => loss.id === lossId).label;
    setStatus(`Loading ${label}…`);
    try {
      currentData = await loadLoss(lossId);
      renderTable();
      renderSelectedCell();
      setStatus("Precomputed GPU diagnostic · select any table cell.");
    } catch (error) {
      setStatus(`Could not load the selected diagnostic: ${error.message}`, true);
    }
  }

  function renderTable() {
    const counts = currentData.cells.map((cell) => cell.target_directed_count);
    const best = Math.max(...counts);
    cellButtons.forEach((button) => {
      const cell = currentData.cells[Number(button.dataset.cell)];
      button.innerHTML = `${cell.target_directed_percentage.toFixed(1)}%<span>${cell.target_directed_count}/120</span>`;
      button.disabled = false;
      button.parentElement.classList.toggle("best", cell.target_directed_count === best);
      button.parentElement.classList.toggle("selected", cell.index === selectedCell);
      button.setAttribute("aria-pressed", String(cell.index === selectedCell));
      button.setAttribute(
        "aria-label",
        `${cell.onset_label} onset, ${cell.propagation_label} propagation: ${cell.target_directed_percentage.toFixed(1)} percent target-directed. Show landscape.`,
      );
    });
    const label = lossLabels[currentData.loss.id] || currentData.loss.label;
    caption.textContent = `Target-directed ${label} descent directions by onset-placement and propagation method.`;
  }

  function selectCell(index) {
    selectedCell = index;
    renderTable();
    renderSelectedCell();
  }

  function prepareCanvas() {
    const cssWidth = Math.max(320, canvas.clientWidth || 760);
    const cssHeight = cssWidth * (820 / 760);
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = Math.round(cssWidth * ratio);
    canvas.height = Math.round(cssHeight * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { width: cssWidth, height: cssHeight };
  }

  function drawArrow(x, y, dx, dy, length) {
    const magnitude = Math.hypot(dx, dy);
    if (!Number.isFinite(magnitude) || magnitude === 0) return;
    const ux = dx / magnitude;
    const uy = dy / magnitude;
    const x0 = x - 0.5 * length * ux;
    const y0 = y - 0.5 * length * uy;
    const x1 = x + 0.5 * length * ux;
    const y1 = y + 0.5 * length * uy;
    const head = Math.max(3.2, length * 0.28);
    const wing = head * 0.62;
    context.beginPath();
    context.moveTo(x0, y0);
    context.lineTo(x1, y1);
    context.stroke();
    context.beginPath();
    context.moveTo(x1, y1);
    context.lineTo(x1 - head * ux + wing * uy, y1 - head * uy - wing * ux);
    context.lineTo(x1 - head * ux - wing * uy, y1 - head * uy + wing * ux);
    context.closePath();
    context.fill();
  }

  function drawLandscape(cell) {
    const { width, height } = prepareCanvas();
    const left = Math.max(58, width * 0.095);
    const right = width - Math.max(18, width * 0.03);
    const top = Math.max(20, width * 0.035);
    const bottom = height - Math.max(94, width * 0.135);
    const plotWidth = right - left;
    const plotHeight = bottom - top;
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#fff";
    context.fillRect(0, 0, width, height);

    const tileWidth = plotWidth / GRID_POINTS;
    const tileHeight = plotHeight / GRID_POINTS;
    for (let pitch = 0; pitch < GRID_POINTS; pitch += 1) {
      for (let onset = 0; onset < GRID_POINTS; onset += 1) {
        const value = cell.normalized_loss[pitch * GRID_POINTS + onset];
        context.fillStyle = magma(value);
        context.fillRect(
          left + onset * tileWidth,
          top + (GRID_POINTS - 1 - pitch) * tileHeight,
          tileWidth + 0.6,
          tileHeight + 0.6,
        );
      }
    }

    context.save();
    context.beginPath();
    context.rect(left, top, plotWidth, plotHeight);
    context.clip();
    context.strokeStyle = "#fff";
    context.fillStyle = "#fff";
    context.lineWidth = Math.max(0.85, width / 760);
    const arrowLength = Math.max(10, Math.min(plotWidth, plotHeight) / 30);
    cell.arrows.forEach(([pitch, onset, dPitch, dOnset]) => {
      const x = left + onset * plotWidth;
      const y = bottom - pitch * plotHeight;
      drawArrow(x, y, dOnset, -dPitch, arrowLength);
    });
    context.restore();

    const targetX = left + 0.5 * plotWidth;
    const targetY = top + 0.5 * plotHeight;
    const cross = Math.max(6, width / 92);
    context.strokeStyle = "#fff59d";
    context.lineWidth = Math.max(2, width / 330);
    context.beginPath();
    context.moveTo(targetX - cross, targetY);
    context.lineTo(targetX + cross, targetY);
    context.moveTo(targetX, targetY - cross);
    context.lineTo(targetX, targetY + cross);
    context.stroke();

    context.strokeStyle = "#172036";
    context.lineWidth = 1;
    context.strokeRect(left, top, plotWidth, plotHeight);
    context.fillStyle = "#172036";
    context.font = `${Math.max(11, width / 62)}px Inter, system-ui, sans-serif`;
    context.textAlign = "center";
    context.textBaseline = "top";
    [[0, "−0.8"], [0.5, "0"], [1, "+0.8"]].forEach(([position, label]) => {
      const x = left + position * plotWidth;
      context.beginPath();
      context.moveTo(x, bottom);
      context.lineTo(x, bottom + 5);
      context.stroke();
      context.fillText(label, x, bottom + 8);
    });
    context.textAlign = "right";
    context.textBaseline = "middle";
    [[0, "−1"], [0.5, "0"], [1, "+1"]].forEach(([position, label]) => {
      const y = bottom - position * plotHeight;
      context.beginPath();
      context.moveTo(left - 5, y);
      context.lineTo(left, y);
      context.stroke();
      context.fillText(label, left - 9, y);
    });
    context.textAlign = "center";
    context.textBaseline = "top";
    context.fillText("Time shift (s)", left + 0.5 * plotWidth, bottom + 31);
    context.save();
    context.translate(Math.max(13, left * 0.22), top + 0.5 * plotHeight);
    context.rotate(-Math.PI / 2);
    context.fillText("f₀ shift (octaves)", 0, 0);
    context.restore();

    const barTop = height - Math.max(34, width * 0.047);
    const barHeight = Math.max(9, width / 72);
    const gradient = context.createLinearGradient(left, 0, right, 0);
    magmaStops.forEach(([position, rgb]) => gradient.addColorStop(position, `rgb(${rgb.join(",")})`));
    context.fillStyle = gradient;
    context.fillRect(left, barTop, plotWidth, barHeight);
    context.strokeStyle = "#172036";
    context.strokeRect(left, barTop, plotWidth, barHeight);
    context.font = `${Math.max(9, width / 76)}px Inter, system-ui, sans-serif`;
    context.textBaseline = "bottom";
    context.textAlign = "left";
    context.fillStyle = "#172036";
    context.fillText("0", left, barTop - 2);
    context.textAlign = "right";
    context.fillText("1", right, barTop - 2);
    context.textAlign = "center";
    context.textBaseline = "top";
    context.fillText("Loss (normalised)", left + 0.5 * plotWidth, barTop + barHeight + 4);
  }

  function renderSelectedCell() {
    if (!currentData) return;
    const cell = currentData.cells[selectedCell];
    const label = lossLabels[currentData.loss.id] || currentData.loss.label;
    title.textContent = label;
    renderer.textContent = `${cell.onset_label} onset placement · ${cell.propagation_label} propagation`;
    score.textContent = `${cell.target_directed_percentage.toFixed(1)}% · ${cell.target_directed_count}/120`;
    score.setAttribute(
      "aria-label",
      `${cell.target_directed_count} of 120 directions point towards the target`,
    );
    canvas.setAttribute(
      "aria-label",
      `${label} landscape for ${cell.onset_label} onset placement and ${cell.propagation_label} propagation, with ${cell.target_directed_count} of 120 directions pointing towards the target.`,
    );
    drawLandscape(cell);
  }

  cellButtons.forEach((button) => {
    button.addEventListener("click", () => selectCell(Number(button.dataset.cell)));
  });

  let resizeFrame;
  window.addEventListener("resize", () => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(renderSelectedCell);
  });

  fetchJson(`${DATA_ROOT}manifest.json`)
    .then((value) => {
      manifest = value;
      buildLossSelector();
      return selectLoss(selectedLoss);
    })
    .catch((error) => {
      setStatus(`Could not load renderer-screen data: ${error.message}`, true);
    });
})();
