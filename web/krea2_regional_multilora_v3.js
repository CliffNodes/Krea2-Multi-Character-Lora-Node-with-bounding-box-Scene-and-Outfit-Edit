// Krea2 Regional Multi-LoRA v3 + Ref Lock (By Fedor) - dynamic region rows
// with per-region reference images.
//
// Per region: enable toggle + LoRA combo + strength + a "load ref" row that
// uploads an image to ComfyUI's input folder and shows an INLINE THUMBNAIL,
// so you can see exactly which image each LoRA references. The filename is
// stored in regions_json ("ref_image"), so save/load and the API round-trip.
//
// Auto-sync: region row count follows the number of boxes drawn in a wired
// bounding-box builder (same as v1).

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "Krea2RegionalMultiLoRAV3";
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
    console.warn("[Krea2RegionalMultiLoRAV3] could not fetch lora list:", e);
  }
  return LORA_LIST;
}

function defaultRegion() {
  return { lora: "None", strength: 1.1, enable: true, ref_image: "", ref_enable: true };
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
// Reference image upload + thumbnail cache
// ---------------------------------------------------------------------------

const THUMB_CACHE = {}; // filename -> HTMLImageElement

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
    console.error("[Krea2RegionalMultiLoRAV3] upload failed:", resp.status);
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
      delete THUMB_CACHE[name]; // force fresh load
      rebuildRows(node);
    }
  };
  input.click();
}

// Custom widget: thumbnail + filename + click-to-load + X-to-clear.
function makeRefWidget(node, idx, region) {
  const w = {
    type: "K2V3REF",
    name: `region ${idx + 1} ref`,
    value: region.ref_image || "",
    serialize: false,
    options: { serialize: false },
    computeSize(width) {
      return [width, this.value ? THUMB_H + 8 : 22];
    },
    draw(ctx, drawNode, widgetWidth, y, H) {
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
        const short = this.value.length > 26 ? "…" + this.value.slice(-25) : this.value;
        ctx.fillText(`ref ${idx + 1}: ${short}`, margin + this.__thumbW + 12, y + 6 + ih / 2 - 2);
        ctx.fillStyle = "#999";
        ctx.font = "10px Arial";
        ctx.fillText("(click image to replace)", margin + this.__thumbW + 12, y + 6 + ih / 2 + 12);
        // clear "X"
        ctx.fillStyle = "#c66";
        ctx.font = "bold 12px Arial";
        ctx.fillText("✕", margin + w_ - 16, y + 16);
      } else {
        ctx.fillStyle = "#bbb";
        ctx.font = "11px Arial";
        ctx.textAlign = "center";
        ctx.fillText(`📷 load ref image for region ${idx + 1}`, margin + w_ / 2, y + 15);
        ctx.textAlign = "left";
      }
      ctx.restore();
    },
    mouse(event, pos, mNode) {
      const evtType = event.type === "pointerdown" || event.type === "mousedown";
      if (!evtType) return false;
      if (this.value) {
        // X zone = clear (right 30px)
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
// Bbox auto-sync (same heuristic as v1)
// ---------------------------------------------------------------------------

// Read the builder's LIVE box array (`_boxes`) - updates immediately on
// create/delete and is length 0 when the last box is removed (the serialized
// STRING widget becomes "" at zero, which is why parsing it could never sync
// down). Falls back to a box-array STRING widget for other builders.
function getBboxCount(node) {
  const bboxInput = node.inputs?.find((i) => i.name === "bboxes");
  if (!bboxInput || bboxInput.link == null) return null;
  const linkInfo = node.graph?.links?.[bboxInput.link];
  if (!linkInfo) return null;
  const srcNode = node.graph?.getNodeById(linkInfo.origin_id);
  if (!srcNode) return null;

  if (Array.isArray(srcNode._boxes)) {
    return srcNode._boxes.length;
  }

  for (const w of srcNode.widgets || []) {
    if (typeof w.value !== "string") continue;
    const s = w.value.trim();
    if (s === "") continue;
    try {
      const parsed = JSON.parse(s);
      if (
        Array.isArray(parsed) &&
        parsed.length > 0 &&
        typeof parsed[0] === "object" &&
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

// Single sync entry point with a load-race guard (don't clear rows to zero
// while a connected builder is still restoring its boxes on graph load).
function checkAndSync(node) {
  const count = getBboxCount(node);
  if (count === null) return;
  if (count === node.__k2lastBboxCount) return;
  if (
    count === 0 &&
    readRegions(node).length > 0 &&
    node.__k2loadGuardUntil &&
    Date.now() < node.__k2loadGuardUntil
  ) {
    return;
  }
  node.__k2lastBboxCount = count;
  syncRegionCount(node, count);
}

// Global backstop: builder edits (create/move on mouseup, delete on Del/Backspace)
// don't reliably repaint our node, so re-check on those events. Once per page.
function installGlobalResync(app) {
  if (window.__k2v3RegionSyncHooked) return;
  window.__k2v3RegionSyncHooked = true;
  const resyncAll = () => {
    const nodes = app.graph?._nodes || [];
    for (const n of nodes) {
      if (n.type === NODE_TYPE) checkAndSync(n);
    }
  };
  window.addEventListener("mouseup", () => setTimeout(resyncAll, 0), true);
  window.addEventListener(
    "keyup",
    (e) => {
      if (e.key === "Delete" || e.key === "Backspace") setTimeout(resyncAll, 0);
    },
    true
  );
}

// ---------------------------------------------------------------------------
// Row rendering
// ---------------------------------------------------------------------------

function rebuildRows(node) {
  if (node.widgets) {
    node.widgets = node.widgets.filter((w) => !w.__k2region);
  }

  const regions = readRegions(node);

  regions.forEach((region, idx) => {
    const enableW = node.addWidget(
      "toggle",
      `region ${idx + 1} enabled`,
      region.enable !== false,
      (v) => {
        const r = readRegions(node);
        if (r[idx]) { r[idx].enable = v; writeRegions(node, r); }
      },
      { on: "on", off: "off" }
    );
    markTransient(enableW);

    const loraW = node.addWidget(
      "combo",
      `region ${idx + 1} lora`,
      region.lora || "None",
      (v) => {
        const r = readRegions(node);
        if (r[idx]) { r[idx].lora = v; writeRegions(node, r); }
      },
      { values: LORA_LIST }
    );
    markTransient(loraW);

    const strW = node.addWidget(
      "number",
      `region ${idx + 1} strength`,
      typeof region.strength === "number" ? region.strength : 1.1,
      (v) => {
        const r = readRegions(node);
        if (r[idx]) { r[idx].strength = v; writeRegions(node, r); }
      },
      { min: -10.0, max: 10.0, step: 0.1, precision: 2 }
    );
    markTransient(strW);

    // Reference image row: upload + inline thumbnail.
    if (node.addCustomWidget) {
      node.addCustomWidget(makeRefWidget(node, idx, region));
    } else {
      node.widgets.push(makeRefWidget(node, idx, region));
    }

    // Per-row "use reference image" toggle - only once an image is loaded.
    if (region.ref_image) {
      const refEnW = node.addWidget(
        "toggle",
        `region ${idx + 1} use reference`,
        region.ref_enable !== false,
        (v) => {
          const r = readRegions(node);
          if (r[idx]) { r[idx].ref_enable = v; writeRegions(node, r); }
        },
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
// Extension registration
// ---------------------------------------------------------------------------

app.registerExtension({
  name: "krea2.RegionalMultiLoRAV3",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;
    await ensureLoraList();
    installGlobalResync(app);

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

      rebuildRows(this);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (o) {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      this.__k2lastBboxCount = null;
      this.__k2loadGuardUntil = Date.now() + 2500;
      setTimeout(() => rebuildRows(this), 0);
      return r;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
      const r = onConnectionsChange
        ? onConnectionsChange.apply(this, arguments)
        : undefined;
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
});
