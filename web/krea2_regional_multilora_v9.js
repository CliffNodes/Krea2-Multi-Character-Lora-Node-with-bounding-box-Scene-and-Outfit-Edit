// Krea2 Regional Multi-LoRA v9 — V2 LoRA+mold likeness always; krea2edit opt-in.
// Per-region rows: LoRA dropdown, strength stepper, enable, portrait prompt,
// and in-node Load Ref upload (regions_json.ref_image). Standalone LoadImage
// plates wire to extra_ref_1/2 for editing anything. No auto-portraits.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// V10 subclasses V9 (same INPUT_TYPES shape + a few tail widgets), so the
// exact same frontend treatment applies: LoRA picker rows, migration, type
// coercion, range clamping. Gating on V9 only left V10 nodes with a bare
// regions_json textarea and no widget repair, which is why V10 saves loaded
// with shifted values and min-violation errors on compose_steps / portrait_steps.
const NODE_TYPES = new Set([
  "Krea2RegionalMultiLoRAV9",
  "Krea2RegionalMultiLoRAV10",
  "Krea2RegionalMultiLoRAV12",
]);
const isKrea2RegionalNode = (name) => NODE_TYPES.has(name);
const JSON_WIDGET = "regions_json";
const THUMB_H = 54;

let LORA_LIST = ["None"];

async function ensureLoraList() {
  if (LORA_LIST.length > 1) return LORA_LIST;
  try {
    const resp = await api.fetchApi("/object_info/LoraLoader");
    const info = await resp.json();
    const names = info?.LoraLoader?.input?.required?.lora_name?.[0];
    if (Array.isArray(names) && names.length) {
      LORA_LIST = ["None", ...names.filter((n) => n !== "None")];
    }
  } catch (e) {
    console.warn("[Krea2RegionalMultiLoRAV9] could not fetch lora list:", e);
  }
  return LORA_LIST;
}

function defaultRegion() {
  return {
    lora: "None", strength: 1.0, enable: true,
    ref_image: "", ref_enable: true, prompt: "", portrait: false,
  };
}

function readRegions(node) {
  const w = node.widgets?.find((x) => x.name === JSON_WIDGET);
  if (!w) return [];
  try {
    const parsed = JSON.parse(w.value || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    return [];
  }
}

function writeRegions(node, regions) {
  const w = node.widgets?.find((x) => x.name === JSON_WIDGET);
  if (!w) return;
  w.value = JSON.stringify(regions, null, 2);
  if (w.inputEl) w.inputEl.value = w.value;
}

function markTransient(w) {
  w.__k2region = true;
  w.serialize = false;
  if (!w.options) w.options = {};
  w.options.serialize = false;
  return w;
}

// ---------------------------------------------------------------------------
// Accelerated strength stepper (rgthree-style: faster when clicking/holding)
// ---------------------------------------------------------------------------

const STRENGTH_MIN = -10.0;
const STRENGTH_MAX = 10.0;
const STRENGTH_BASE_STEP = 0.05;

function clampStrength(v) {
  return Math.max(STRENGTH_MIN, Math.min(STRENGTH_MAX, Math.round(v * 100) / 100));
}

function makeStrengthWidget(node, idx, region) {
  const ctrl = { repeatTimer: null, repeatCount: 0, lastStepAt: 0, streak: 0, holdDir: 0 };

  function clearRepeat() {
    if (ctrl.repeatTimer != null) {
      clearTimeout(ctrl.repeatTimer);
      ctrl.repeatTimer = null;
    }
    ctrl.repeatCount = 0;
    ctrl.holdDir = 0;
  }

  function stepStrength(direction, fromHold = false) {
    const now = Date.now();
    if (!fromHold) {
      if (now - ctrl.lastStepAt < 220) ctrl.streak = Math.min(ctrl.streak + 1, 10);
      else ctrl.streak = 0;
    }
    ctrl.lastStepAt = now;

    const clickMult = fromHold
      ? 1 + Math.min(ctrl.repeatCount, 24) * 0.12
      : 1 + ctrl.streak * 0.35;
    const delta = STRENGTH_BASE_STEP * clickMult * direction;

    const r = readRegions(node);
    if (!r[idx]) return;
    r[idx].strength = clampStrength((r[idx].strength ?? 1.0) + delta);
    writeRegions(node, r);
    w.value = r[idx].strength;
    node.setDirtyCanvas(true, true);
  }

  function startHold(direction) {
    clearRepeat();
    ctrl.holdDir = direction;
    stepStrength(direction, false);
    ctrl.repeatCount = 0;
    let delay = 320;
    const tick = () => {
      if (ctrl.holdDir !== direction) return;
      ctrl.repeatCount += 1;
      stepStrength(direction, true);
      delay = Math.max(35, delay * 0.86);
      ctrl.repeatTimer = setTimeout(tick, delay);
    };
    ctrl.repeatTimer = setTimeout(tick, delay);
  }

  const w = {
    type: "K2V7STR",
    name: `region ${idx + 1} strength`,
    value: typeof region.strength === "number" ? region.strength : 1.0,
    serialize: false,
    options: { serialize: false },
    computeSize(width) {
      return [width, 22];
    },
    draw(ctx, drawNode, widgetWidth, y) {
      const margin = 12;
      const rowW = widgetWidth - margin * 2;
      const h = 20;
      const midY = y + h / 2;
      const arrowW = 18;
      const valW = 52;
      const decX = margin + rowW - arrowW * 2 - valW - 6;
      const valX = decX + arrowW;
      const incX = valX + valW;

      ctx.save();
      ctx.fillStyle = "#bbb";
      ctx.font = "11px Arial";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(`region ${idx + 1} strength`, margin, midY);

      ctx.fillStyle = "#353535";
      ctx.strokeStyle = "#555";
      ctx.beginPath();
      ctx.roundRect(decX - 2, y + 1, arrowW * 2 + valW + 4, h, 5);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#ddd";
      ctx.textAlign = "center";
      ctx.fillText("\u2212", decX + arrowW / 2, midY);
      ctx.fillText(String(w.value), valX + valW / 2, midY);
      ctx.fillText("+", incX + arrowW / 2, midY);
      ctx.restore();

      this.__decBounds = [decX, y + 1, arrowW, h];
      this.__valBounds = [valX, y + 1, valW, h];
      this.__incBounds = [incX, y + 1, arrowW, h];
    },
    mouse(event, pos, mNode) {
      const inBox = (b) =>
        b && pos[0] >= b[0] && pos[0] <= b[0] + b[2] && pos[1] >= b[1] && pos[1] <= b[1] + b[3];
      if (event.type === "pointerdown" || event.type === "mousedown") {
        if (inBox(this.__decBounds)) { startHold(-1); return true; }
        if (inBox(this.__incBounds)) { startHold(1); return true; }
        if (inBox(this.__valBounds)) {
          app.canvas.prompt("Strength", w.value, (v) => {
            const r = readRegions(mNode);
            if (!r[idx]) return;
            r[idx].strength = clampStrength(Number(v));
            writeRegions(mNode, r);
            w.value = r[idx].strength;
            mNode.setDirtyCanvas(true, true);
          }, event);
          return true;
        }
      }
      if (event.type === "pointerup" || event.type === "mouseup") clearRepeat();
      return false;
    },
  };
  markTransient(w);
  return w;
}

// ---------------------------------------------------------------------------
// Reference image upload + thumbnail cache
// ---------------------------------------------------------------------------

const THUMB_CACHE = {};

function thumbFor(name, node) {
  if (!name) return null;
  let img = THUMB_CACHE[name];
  if (!img) {
    img = new Image();
    const slash = name.lastIndexOf("/");
    const subfolder = slash >= 0 ? name.slice(0, slash) : "";
    const fname = slash >= 0 ? name.slice(slash + 1) : name;
    img.src = api.apiURL(
      `/view?filename=${encodeURIComponent(fname)}&type=input&subfolder=${encodeURIComponent(subfolder)}&rand=${Math.random()}`
    );
    img.onload = () => node.setDirtyCanvas(true, true);
    THUMB_CACHE[name] = img;
  }
  return img.complete && img.naturalWidth ? img : null;
}

async function uploadRefImage(file) {
  const body = new FormData();
  body.append("image", file);
  body.append("type", "input");
  const resp = await api.fetchApi("/upload/image", { method: "POST", body });
  if (resp.status !== 200) {
    console.error("[Krea2RegionalMultiLoRAV9] upload failed:", resp.status);
    return null;
  }
  const data = await resp.json();
  return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

function pickAndUploadRef(node, idx) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/png,image/jpeg,image/webp,image/bmp";
  input.onchange = async () => {
    const file = input.files?.[0];
    if (!file) return;
    const name = await uploadRefImage(file);
    if (!name) return;
    const r = readRegions(node);
    if (r[idx]) {
      r[idx].ref_image = name;
      if (r[idx].ref_enable === undefined) r[idx].ref_enable = true;
      writeRegions(node, r);
      delete THUMB_CACHE[name];
      rebuildRows(node);
    }
  };
  input.click();
}

function makeRefWidget(node, idx, region) {
  const w = {
    type: "K2V7REF",
    name: `region ${idx + 1} ref`,
    value: region.ref_image || "",
    serialize: false,
    options: { serialize: false },
    computeSize(width) {
      return [width, this.value ? THUMB_H + 8 : 22];
    },
    draw(ctx, drawNode, widgetWidth, y) {
      const margin = 12;
      const w_ = widgetWidth - margin * 2;
      ctx.save();
      ctx.fillStyle = "#353535";
      ctx.strokeStyle = "#555";
      const h = this.value ? THUMB_H + 4 : 18;
      ctx.beginPath();
      ctx.roundRect(margin, y + 2, w_, h, 6);
      ctx.fill();
      ctx.stroke();

      if (this.value) {
        const img = thumbFor(this.value, drawNode);
        const ih = THUMB_H - 4;
        if (img) {
          const iw = Math.min(ih * (img.naturalWidth / img.naturalHeight), 96);
          ctx.save();
          ctx.beginPath();
          ctx.roundRect(margin + 4, y + 6, iw, ih, 4);
          ctx.clip();
          ctx.drawImage(img, margin + 4, y + 6, iw, ih);
          ctx.restore();
          this.__thumbW = iw;
        } else {
          ctx.fillStyle = "#666";
          ctx.fillRect(margin + 4, y + 6, 48, ih);
          this.__thumbW = 48;
        }
        ctx.fillStyle = "#ddd";
        ctx.font = "11px Arial";
        const short = this.value.length > 26 ? "\u2026" + this.value.slice(-25) : this.value;
        ctx.fillText(`ref ${idx + 1}: ${short}`, margin + this.__thumbW + 12, y + 6 + ih / 2 - 2);
        ctx.fillStyle = "#999";
        ctx.font = "10px Arial";
        ctx.fillText("(click image to replace)", margin + this.__thumbW + 12, y + 6 + ih / 2 + 12);
        ctx.fillStyle = "#c66";
        ctx.font = "bold 12px Arial";
        ctx.fillText("\u2715", margin + w_ - 16, y + 16);
      } else {
        ctx.fillStyle = "#bbb";
        ctx.font = "11px Arial";
        ctx.textAlign = "center";
        ctx.fillText(`\uD83D\uDCF7 load ref image for region ${idx + 1} (optional)`, margin + w_ / 2, y + 15);
        ctx.textAlign = "left";
      }
      ctx.restore();
    },
    mouse(event, pos, mNode) {
      const evtType = event.type === "pointerdown" || event.type === "mousedown";
      if (!evtType) return false;
      if (this.value) {
        if (pos[0] > mNode.size[0] - 42) {
          const r = readRegions(mNode);
          if (r[idx]) {
            r[idx].ref_image = "";
            writeRegions(mNode, r);
            rebuildRows(mNode);
          }
          return true;
        }
      }
      pickAndUploadRef(mNode, idx);
      return true;
    },
  };
  markTransient(w);
  return w;
}

// ---------------------------------------------------------------------------
// Bbox auto-sync (same heuristic as v3-v6)
// ---------------------------------------------------------------------------

function getBboxCount(node) {
  const bboxInput = node.inputs?.find((i) => i.name === "bboxes");
  if (!bboxInput || bboxInput.link == null) return null;
  const linkInfo = node.graph?.links?.[bboxInput.link];
  if (!linkInfo) return null;
  const srcNode = node.graph?.getNodeById(linkInfo.origin_id);
  if (!srcNode) return null;

  if (Array.isArray(srcNode._boxes)) return srcNode._boxes.length;

  for (const w of srcNode.widgets || []) {
    if (typeof w.value !== "string") continue;
    const s = w.value.trim();
    if (s === "") continue;
    try {
      const parsed = JSON.parse(s);
      if (
        Array.isArray(parsed) && parsed.length > 0 && typeof parsed[0] === "object" &&
        ("x" in parsed[0] || "x0" in parsed[0] || "width" in parsed[0] || "w" in parsed[0])
      ) {
        return parsed.length;
      }
    } catch (_) {}
  }
  return null;
}

function syncRegionCount(node, targetCount) {
  const regions = readRegions(node);
  if (regions.length === targetCount) return;
  if (regions.length < targetCount) {
    while (regions.length < targetCount) regions.push(defaultRegion());
  } else {
    regions.splice(targetCount);
  }
  writeRegions(node, regions);
  rebuildRows(node);
}

function checkAndSync(node) {
  const count = getBboxCount(node);
  if (count === null) return;
  if (count === node.__k2lastBboxCount) return;
  if (
    count === 0 && readRegions(node).length > 0 &&
    node.__k2loadGuardUntil && Date.now() < node.__k2loadGuardUntil
  ) {
    return;
  }
  node.__k2lastBboxCount = count;
  syncRegionCount(node, count);
}

function installGlobalResync(app) {
  if (window.__k2v7RegionSyncHooked) return;
  window.__k2v7RegionSyncHooked = true;
  const resyncAll = () => {
    const nodes = app.graph?._nodes || [];
    for (const n of nodes) if (isKrea2RegionalNode(n.type)) checkAndSync(n);
  };
  window.addEventListener("mouseup", () => setTimeout(resyncAll, 0), true);
  window.addEventListener("keyup", (e) => {
    if (e.key === "Delete" || e.key === "Backspace") setTimeout(resyncAll, 0);
  }, true);
}

// ---------------------------------------------------------------------------
// Row rendering
// ---------------------------------------------------------------------------

function rebuildRows(node) {
  if (node.widgets) node.widgets = node.widgets.filter((w) => !w.__k2region);

  const regions = readRegions(node);

  regions.forEach((region, idx) => {
    const enableW = node.addWidget(
      "toggle", `region ${idx + 1} enabled`, region.enable !== false,
      (v) => { const r = readRegions(node); if (r[idx]) { r[idx].enable = v; writeRegions(node, r); } },
      { on: "on", off: "off" }
    );
    markTransient(enableW);

    const loraW = node.addWidget(
      "combo", `region ${idx + 1} lora`, region.lora || "None",
      (v) => { const r = readRegions(node); if (r[idx]) { r[idx].lora = v; writeRegions(node, r); } },
      { values: LORA_LIST }
    );
    markTransient(loraW);

    if (node.addCustomWidget) node.addCustomWidget(makeStrengthWidget(node, idx, region));
    else node.widgets.push(makeStrengthWidget(node, idx, region));

    // Per-region portrait prompt: guides the LoRA-only auto portrait (e.g. "a man").
    const promptW = node.addWidget(
      "text", `region ${idx + 1} portrait prompt`, region.prompt || "",
      (v) => { const r = readRegions(node); if (r[idx]) { r[idx].prompt = v; writeRegions(node, r); } },
      {}
    );
    markTransient(promptW);

    // Per-region portrait opt-in. Only offered when the region has no live
    // reference photo, because a photo already supplies the subject frame and
    // the portrait would never be rendered. Costs an extra render plus a model
    // reload, so it is a deliberate per-LoRA choice: switch it on to inspect how
    // faithful a LoRA is alone, off once you trust it.
    const refLive = !!region.ref_image && region.ref_enable !== false;
    if (!refLive) {
      const portraitW = node.addWidget(
        "toggle", `region ${idx + 1} portrait (slower)`, region.portrait === true,
        (v) => { const r = readRegions(node); if (r[idx]) { r[idx].portrait = v; writeRegions(node, r); } },
        { on: "on", off: "off" }
      );
      markTransient(portraitW);
    }

    // Reference image row: upload + inline thumbnail (a real photo for this region).
    if (node.addCustomWidget) node.addCustomWidget(makeRefWidget(node, idx, region));
    else node.widgets.push(makeRefWidget(node, idx, region));

    if (region.ref_image) {
      const refEnW = node.addWidget(
        "toggle", `region ${idx + 1} use reference`, region.ref_enable !== false,
        (v) => { const r = readRegions(node); if (r[idx]) { r[idx].ref_enable = v; writeRegions(node, r); } },
        { on: "on", off: "off" }
      );
      markTransient(refEnW);
    }

    const rmW = node.addWidget("button", `  remove region ${idx + 1}`, null, () => {
      const r = readRegions(node);
      r.splice(idx, 1);
      writeRegions(node, r);
      rebuildRows(node);
      node.setDirtyCanvas(true, true);
    });
    markTransient(rmW);
  });

  const sz = node.computeSize();
  node.size[1] = Math.max(node.size[1], sz[1]);
  node.setDirtyCanvas(true, true);
}

// ---------------------------------------------------------------------------
// Legacy save migration
// ---------------------------------------------------------------------------
// The first V9 build had edit_lora as an OPTIONAL widget at index 10. It is now
// REQUIRED at index 0 (matching V8), so graphs saved with the old layout load one
// slot out of phase (canvas_width lands in edit_lora, etc.). Pull the LoRA
// filename back to the front so old saves self-correct.

function migrateWidgetsValues(info) {
  const wv = info?.widgets_values;
  if (!Array.isArray(wv) || wv.length < 2) return;
  const isLora = (v) => typeof v === "string" && /\.safetensors$/i.test(v);
  if (isLora(wv[0])) return;
  const idx = wv.findIndex((v, i) => i < 14 && isLora(v));
  if (idx > 0) {
    wv.unshift(wv.splice(idx, 1)[0]);
    console.log("[Krea2RegionalMultiLoRAV9] migrated legacy widget order (edit_lora -> slot 0)");
  }
}

// Stale saves can also carry values outside the current widget bounds
// (e.g. seam_feather 0.66 when max is 0.5). Clamp instead of failing the load.
function clampWidgetRanges(node) {
  for (const w of node.widgets || []) {
    if (w.__k2region || typeof w.value !== "number" || !w.options) continue;
    const { min, max } = w.options;
    if (typeof min === "number" && w.value < min) w.value = min;
    if (typeof max === "number" && w.value > max) w.value = max;
  }
}

// ...and a value of the wrong TYPE entirely, which clamping cannot help with.
//
// Appending widgets to this node has now broken stale graphs three times:
// edit_lora received 1536, seam_feather received 0.66, ref_max_side received
// "0,0,1,1". Every time, a session still holding a pre-append graph mapped its
// saved values onto the new widget list out of phase. The backend rejects the
// type during validation, so no Python ever runs and nothing server-side can
// defend against it. Repair here instead of failing the queue.
function coerceWidgetTypes(node) {
  for (const w of node.widgets || []) {
    if (w.__k2region || w.__k2add) continue;
    const def = w.options?.default;
    if (w.type === "number") {
      if (typeof w.value === "number" && Number.isFinite(w.value)) continue;
      const n = Number(w.value);
      w.value = Number.isFinite(n) ? n : (typeof def === "number" ? def : 0);
      console.warn(`[Krea2RegionalMultiLoRAV9] repaired ${w.name}: `
        + `non-numeric value -> ${w.value}`);
    } else if (w.type === "toggle") {
      if (typeof w.value === "boolean") continue;
      w.value = typeof def === "boolean" ? def : Boolean(w.value);
      console.warn(`[Krea2RegionalMultiLoRAV9] repaired ${w.name} -> ${w.value}`);
    } else if (w.type === "combo" && Array.isArray(w.options?.values)) {
      if (w.options.values.includes(w.value)) continue;
      w.value = w.options.values.includes(def) ? def : w.options.values[0];
      console.warn(`[Krea2RegionalMultiLoRAV9] repaired ${w.name}: `
        + `value not in list -> ${w.value}`);
    }
  }
}

// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------

app.registerExtension({
  name: "krea2.RegionalMultiLoRAV9",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!isKrea2RegionalNode(nodeData.name)) return;
    await ensureLoraList();
    installGlobalResync(app);

    const configure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      try {
        migrateWidgetsValues(info);
      } catch (e) {
        console.warn("[Krea2RegionalMultiLoRAV9] widget migration failed:", e);
      }
      return configure ? configure.apply(this, arguments) : undefined;
    };

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      this.__k2lastBboxCount = null;

      const addBtn = this.addWidget("button", "+ Add Region", null, () => {
        const regions = readRegions(this);
        regions.push(defaultRegion());
        writeRegions(this, regions);
        rebuildRows(this);
      });
      addBtn.__k2add = true;
      // Buttons carry no value; keep it out of widgets_values so it can never
      // become one more slot that shifts a future appended widget.
      addBtn.serialize = false;
      if (addBtn.options) addBtn.options.serialize = false;

      coerceWidgetTypes(this);
      rebuildRows(this);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (o) {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      this.__k2lastBboxCount = null;
      this.__k2loadGuardUntil = Date.now() + 2500;
      coerceWidgetTypes(this);
      clampWidgetRanges(this);
      setTimeout(() => rebuildRows(this), 0);
      return r;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
      const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
      const bboxIdx = this.inputs?.findIndex((i) => i.name === "bboxes");
      if (index === bboxIdx) {
        this.__k2lastBboxCount = null;
        if (connected) this.__k2loadGuardUntil = 0;
        setTimeout(() => checkAndSync(this), 50);
      }
      return r;
    };

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      if (onDrawForeground) onDrawForeground.apply(this, arguments);
      checkAndSync(this);
    };
  },

  // onConfigure is not reached on every load path, so repair here too.
  loadedGraphNode(node) {
    if (!isKrea2RegionalNode(node?.comfyClass) && !isKrea2RegionalNode(node?.type)) return;
    try {
      coerceWidgetTypes(node);
      clampWidgetRanges(node);
    } catch (e) {
      console.warn("[Krea2RegionalMultiLoRAV9] widget repair failed:", e);
    }
  },
});
