# `bioelectricity.net` Integration Package (By Joel Dietz)

This folder (`docs/bioelectricity/`) provides a turnkey integration bundle so our **Self-Sorting Cells, "Platonic Space" & Causal Emergence 2.0** module and our **D'Arcy Thompson 1917 / 10,303-Species Morphospace & Bioelectric Prepattern** discoveries (plus Lean-4-verified *Conus textile* Rule 30 neural-gate proofs) can be dropped straight into **[bioelectricity.net](https://bioelectricity.net)** (*Bioelectricity Nexus*).

---

## 1. Module A: Self-Sorting Cells, "Platonic Space," & Causal Emergence 2.0 (Zhang, Goldstein & Levin 2025 + Hoel 2026)

- **Interactive Gym-Class Cell Simulator & Illustrated Guide (`<iframe />` / Web App):**  
  `https://fractastical.github.io/morpholib/bioelectricity/sorting-morphogenesis.html`
- **4-Page Magazine Edition Illustrated Student's Guide PDF (13-Year-Old Level):**  
  `https://fractastical.github.io/morpholib/zhang_levin_13yo_illustrated_guide.pdf`
- **8-Page Full Academic & Causal Emergence 2.0 Manuscript PDF:**  
  `https://fractastical.github.io/morpholib/zhang_levin_sorting_morphogenesis_manuscript.pdf`
- **Drop-In React / TypeScript Component (`SortingMorphogenesisWidget.tsx`):**  
  [`docs/bioelectricity/SortingMorphogenesisWidget.tsx`](./SortingMorphogenesisWidget.tsx)

### Quick `<iframe />` Embed for `bioelectricity.net`:
```html
<iframe
  src="https://fractastical.github.io/morpholib/bioelectricity/sorting-morphogenesis.html"
  title="Self-Sorting Cells, Platonic Space & Causal Emergence 2.0 — Illustrated Guide & Simulator by Joel Dietz"
  style="width: 100%; height: 920px; border: 1px solid #263859; border-radius: 16px; background: #070b14;"
  loading="lazy"
  allowfullscreen
></iframe>
```

### Quick React / Vite Component Import:
```tsx
import { SortingMorphogenesisWidget } from './components/SortingMorphogenesisWidget';

export function SortingMorphogenesisSection() {
  return <SortingMorphogenesisWidget defaultTab="guide" />;
}
```

---

## 2. Module B: D'Arcy Thompson 1917 & 10,303-Species Bioelectric Morphospace

- **Interactive Web App & `<iframe />` Widget:**  
  `https://fractastical.github.io/morpholib/bioelectricity/`
- **Structured JSON API Feed (Both Modules + Rule 30):**  
  `https://fractastical.github.io/morpholib/bioelectricity/bioelectricity_nexus_feed.json`
- **Drop-In React / TypeScript Component (`InfiniteMorphospaceWidget.tsx`):**  
  [`docs/bioelectricity/InfiniteMorphospaceWidget.tsx`](./InfiniteMorphospaceWidget.tsx)
- **Visual Guide PDF (13-Year-Old Level, Large-Font Separate Diagrams):**  
  `https://fractastical.github.io/morpholib/thompson_1917_visual_guide_for_everyone.pdf`
- **Full Academic Manuscript PDF:**  
  `https://fractastical.github.io/morpholib/thompson_1917_fish_transformations_manuscript.pdf`
