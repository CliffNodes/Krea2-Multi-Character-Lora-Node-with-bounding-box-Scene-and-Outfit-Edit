import { app } from "../../scripts/app.js";

const MODEL_FILE_RE = /\.(?:safetensors|ckpt|pt|pth|bin)$/i;
const MIN_FAMILY_TOKEN_LENGTH = 3;
const FEDOR_NODE_IDS = new Set([
  "Krea2BoxInpaintMask",
  "Krea2ReferenceLock",
  "Krea2ReferenceLockMulti",
  "Krea2RegionCollage",
  "Krea2RegionalDetailer",
  "Krea2RegionalEditPatch",
  "Krea2RegionalMultiLoRA",
  "Krea2RegionalMultiLoRAV3",
  "Krea2RegionalMultiLoRAV4",
  "Krea2RegionalMultiLoRAV5",
  "Krea2RegionalMultiLoRAV6",
  "Krea2RegionalMultiLoRAV7",
  "Krea2RegionalMultiLoRAV8",
  "Krea2RegionalMultiLoRAV9",
  "Krea2RegionalMultiLoRAV10",
  "Krea2RegionalMultiLoRAV12",
  "Krea2RegionalMultiPassSampler",
]);

function brandNode(node) {
  if (!node || !FEDOR_NODE_IDS.has(node.comfyClass)) return;
  const title = String(node.title || node.comfyClass);
  if (!title.startsWith("Fedor Nodes")) {
    node.title = `Fedor Nodes | ${title}`;
    node.setDirtyCanvas?.(true, true);
  }
}

function basename(path) {
  return String(path).replaceAll("\\", "/").split("/").pop() || String(path);
}

function displayTokens(path) {
  const stem = basename(path).replace(/\.[^.]+$/, "");
  return stem
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/([A-Za-z])(\d+)/g, "$1 $2")
    .replace(/(\d+)([A-Za-z])/g, "$1 $2")
    .split(/[\s_.-]+/)
    .filter(Boolean);
}

function normalizedTokens(path) {
  const tokens = displayTokens(path).map((token) => token.toLocaleLowerCase());
  const versionIndex = tokens.lastIndexOf("v");
  if (
    versionIndex >= 0 &&
    versionIndex < tokens.length - 1 &&
    tokens.slice(versionIndex + 1).every((token) => /^\d+$/.test(token))
  ) {
    tokens.splice(versionIndex);
  }
  return tokens;
}

function buildPrefixTrie(entries) {
  const root = { count: 0, children: new Map() };
  for (const entry of entries) {
    let node = root;
    node.count += 1;
    for (const token of entry.tokens) {
      if (!node.children.has(token)) {
        node.children.set(token, { count: 0, children: new Map() });
      }
      node = node.children.get(token);
      node.count += 1;
    }
  }
  return root;
}

function familyFor(entry, trie) {
  let node = trie;
  const family = [];
  for (const token of entry.tokens) {
    const child = node.children.get(token);
    if (!child || child.count < 2) break;
    family.push(token);
    node = child;
  }
  if (
    family.length === 0 ||
    (family.length === 1 && family[0].length < MIN_FAMILY_TOKEN_LENGTH)
  ) {
    return null;
  }
  return family.join(" ");
}

function titleForFamily(family, entries) {
  const matching = entries.find((entry) =>
    entry.tokens.slice(0, family.split(" ").length).join(" ") === family
  );
  if (!matching) return family;
  return matching.display
    .slice(0, family.split(" ").length)
    .join(" ");
}

function createFolder(name, items) {
  const folder = document.createElement("div");
  folder.className = "litemenu-entry fedor-model-family";
  const arrow = document.createElement("span");
  arrow.className = "fedor-model-family-arrow";
  arrow.textContent = "▶";
  const label = document.createElement("span");
  label.textContent = name;
  const count = document.createElement("span");
  count.className = "fedor-model-family-count";
  count.textContent = String(items.length);
  folder.append(arrow, label, count);

  const contents = document.createElement("div");
  contents.className = "fedor-model-family-contents";
  contents.style.display = "none";

  for (const item of items) {
    const fullPath = item.getAttribute("data-value") || "";
    item.textContent = basename(fullPath);
    item.title = fullPath;
    contents.appendChild(item);
  }

  folder.addEventListener("click", (event) => {
    event.stopPropagation();
    const open = contents.style.display !== "none";
    contents.style.display = open ? "none" : "block";
    arrow.textContent = open ? "▶" : "▼";
  });

  return { folder, contents };
}

function organizeModelMenu(menu) {
  if (menu.dataset.fedorModelOrganizer === "done") return;

  const items = [...menu.querySelectorAll(".litemenu-entry[data-value]")]
    .filter((item) => MODEL_FILE_RE.test(item.getAttribute("data-value") || ""));
  if (items.length < 2) return;

  const entries = items.map((item) => {
    const value = item.getAttribute("data-value");
    return {
      item,
      value,
      tokens: normalizedTokens(value),
      display: displayTokens(value),
    };
  });
  const trie = buildPrefixTrie(entries);
  const groups = new Map();
  const ungrouped = [];

  for (const entry of entries) {
    const family = familyFor(entry, trie);
    if (!family) {
      ungrouped.push(entry.item);
      continue;
    }
    if (!groups.has(family)) groups.set(family, []);
    groups.get(family).push(entry.item);
  }

  if (groups.size === 0) return;
  menu.dataset.fedorModelOrganizer = "done";

  const anchor = items[0];
  const host = anchor.parentElement || menu;
  const fragment = document.createDocumentFragment();
  const sortedGroups = [...groups.entries()].sort(([a], [b]) =>
    a.localeCompare(b)
  );
  const marker = document.createComment("fedor-model-groups");
  host.insertBefore(marker, anchor);

  for (const [family, familyItems] of sortedGroups) {
    const { folder, contents } = createFolder(
      titleForFamily(family, entries),
      familyItems
    );
    fragment.append(folder, contents);
  }

  for (const item of ungrouped) {
    const fullPath = item.getAttribute("data-value") || "";
    item.textContent = basename(fullPath);
    item.title = fullPath;
    fragment.appendChild(item);
  }

  host.insertBefore(fragment, marker);
  marker.remove();
}

app.registerExtension({
  name: "FedorNodes.ModelOrganizer",
  nodeCreated(node) {
    brandNode(node);
  },
  loadedGraphNode(node) {
    // Saved workflows persist custom node titles, which override
    // NODE_DISPLAY_NAME_MAPPINGS. Rebrand those existing titles on load.
    brandNode(node);
  },
  setup() {
    const style = document.createElement("style");
    style.textContent = `
      .fedor-model-family {
        align-items: center;
        display: flex;
        gap: 5px;
        font-weight: 600;
        opacity: 0.9;
      }
      .fedor-model-family-arrow {
        display: inline-block;
        width: 13px;
      }
      .fedor-model-family-count {
        margin-left: auto;
        opacity: 0.55;
        padding-right: 5px;
      }
      .fedor-model-family-contents > .litemenu-entry {
        padding-left: 22px !important;
      }
      .litecontextmenu:has(input:not(:placeholder-shown))
        .fedor-model-family {
        display: none;
      }
      .litecontextmenu:has(input:not(:placeholder-shown))
        .fedor-model-family-contents {
        display: block !important;
      }
    `;
    document.head.appendChild(style);

    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        for (const added of mutation.addedNodes) {
          if (!(added instanceof HTMLElement)) continue;
          const menus = added.classList.contains("litecontextmenu")
            ? [added]
            : [...added.querySelectorAll(".litecontextmenu")];
          for (const menu of menus) {
            requestAnimationFrame(() => organizeModelMenu(menu));
          }
        }
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });
  },
});
