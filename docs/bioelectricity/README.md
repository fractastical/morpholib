# `bioelectricity.net` Integration Package & Queryable Integration Guide (By Joel Dietz)

This folder (`docs/bioelectricity/`) provides a turnkey integration bundle and a **Queryable Integration Guide** (with URL query parameters, `window.postMessage` RPC control, OpenAPI 3.1 schema, and an interactive Embed Builder) so our **Self-Sorting Cells, "Platonic Space" & Causal Emergence 2.0** module and our **D'Arcy Thompson 1917 / 10,303-Species Morphospace & Bioelectric Prepattern** discoveries can be queried and dropped straight into **[bioelectricity.net](https://bioelectricity.net)** (*Bioelectricity Nexus*).

---

## 1. Queryable Integration Guide & API Endpoints

- **Interactive Queryable Integration Guide & Embed Builder (Web UI):**  
  `https://fractastical.github.io/morpholib/bioelectricity/integration-guide.html`  
  *(Local file: [`docs/bioelectricity/integration-guide.html`](./integration-guide.html))*
- **Machine-Queryable Integration Spec (`integration-guide.json`):**  
  `https://fractastical.github.io/morpholib/bioelectricity/integration-guide.json`  
  *(Local file: [`docs/bioelectricity/integration-guide.json`](./integration-guide.json))*
- **OpenAPI 3.1 Specification (`openapi.json`):**  
  `https://fractastical.github.io/morpholib/bioelectricity/openapi.json`  
  *(Local file: [`docs/bioelectricity/openapi.json`](./openapi.json))*
- **Structured Discovery Feed (`bioelectricity_nexus_feed.json`):**  
  `https://fractastical.github.io/morpholib/bioelectricity/bioelectricity_nexus_feed.json`  
  *(Local file: [`docs/bioelectricity/bioelectricity_nexus_feed.json`](./bioelectricity_nexus_feed.json))*

### Supported URL Query Parameters on `sorting-morphogenesis.html`

You can deep-link or embed any specific view or experiment using query parameters:
- `?tab=guide13` | `sim` | `magazine` | `ce2` | `embed`
- `?scenario=rulers` | `insertion_on` | `insertion_off` | `unique_wave` | `c2_anticluster` | `c6_cliques`
- `?autoplay=1` — Immediately starts sorting on load.
- `?compact=1` — Hides the top hero banner for clean inline embedding inside an article card.

**Example Queryable Deep-Links:**
- **C = 2 Anti-Clustering Smoking Gun (Autoplay + Compact Mode):**  
  `https://fractastical.github.io/morpholib/bioelectricity/sorting-morphogenesis.html?tab=sim&scenario=c2_anticluster&autoplay=1&compact=1`
- **"Two Rulers" Optical Illusion (Autoplay):**  
  `https://fractastical.github.io/morpholib/bioelectricity/sorting-morphogenesis.html?tab=sim&scenario=rulers&autoplay=1`
- **4-Page Magazine Spread Preview (Compact Mode):**  
  `https://fractastical.github.io/morpholib/bioelectricity/sorting-morphogenesis.html?tab=magazine&compact=1`

### `window.postMessage` RPC Control & Live Telemetry

Parent pages on `bioelectricity.net` can control the embedded `<iframe />` programmatically and receive live telemetry after every cell swap:

```javascript
// Send a command into the iframe:
iframeElement.contentWindow.postMessage({
  type: 'MORPHOLIB_SET_SCENARIO',
  scenario: 'c2_anticluster',
  autoplay: true
}, '*');

// Listen for live swap telemetry from the iframe:
window.addEventListener('message', (event) => {
  if (event.data?.type === 'MORPHOLIB_SIM_STATE') {
    console.log('Swaps:', event.data.swaps,
                'Ruler 1 I(v):', event.data.ruler1_inversions_I,
                'Ruler 2 E(v):', event.data.ruler2_adj_error_E,
                'Clustering:', event.data.algotype_clustering);
  }
});
```

---

## 2. Module A: Self-Sorting Cells, "Platonic Space," & Causal Emergence 2.0

- **Interactive Gym-Class Cell Simulator & Illustrated Guide (`<iframe />` / Web App):**  
  `https://fractastical.github.io/morpholib/bioelectricity/sorting-morphogenesis.html`
- **4-Page Magazine Edition Illustrated Student's Guide PDF (13-Year-Old Level):**  
  `https://fractastical.github.io/morpholib/zhang_levin_13yo_illustrated_guide.pdf`
- **8-Page Full Academic & Causal Emergence 2.0 Manuscript PDF:**  
  `https://fractastical.github.io/morpholib/zhang_levin_sorting_morphogenesis_manuscript.pdf`
- **Drop-In React / TypeScript Component (`SortingMorphogenesisWidget.tsx`):**  
  [`docs/bioelectricity/SortingMorphogenesisWidget.tsx`](./SortingMorphogenesisWidget.tsx)

---

## 3. Module B: D'Arcy Thompson 1917 & 10,303-Species Bioelectric Morphospace

- **Interactive Web App & `<iframe />` Widget:**  
  `https://fractastical.github.io/morpholib/bioelectricity/`
- **Drop-In React / TypeScript Component (`InfiniteMorphospaceWidget.tsx`):**  
  [`docs/bioelectricity/InfiniteMorphospaceWidget.tsx`](./InfiniteMorphospaceWidget.tsx)
- **Visual Guide PDF (13-Year-Old Level):**  
  `https://fractastical.github.io/morpholib/thompson_1917_visual_guide_for_everyone.pdf`
- **Full Academic Manuscript PDF:**  
  `https://fractastical.github.io/morpholib/thompson_1917_fish_transformations_manuscript.pdf`
