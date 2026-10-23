# `bioelectricity.net` Integration Package (By Joel Dietz)

This folder (`docs/bioelectricity/`) provides a turnkey integration bundle so our D'Arcy Thompson 1917 / 10,303-species Morphospace & Bioelectric Prepattern discoveries (and Lean-4-verified *Conus textile* Rule 30 neural-gate proofs) can be dropped straight into **[bioelectricity.net](https://bioelectricity.net)** (*Bioelectricity Nexus*).

## Live Hosted Endpoints (via GitHub Pages)

- **Interactive Web App & `<iframe />` Widget:**  
  `https://fractastical.github.io/morpholib/bioelectricity/`
- **Structured JSON API Feed:**  
  `https://fractastical.github.io/morpholib/bioelectricity/bioelectricity_nexus_feed.json`
- **Drop-In React / TypeScript Component (`InfiniteMorphospaceWidget.tsx`):**  
  [`docs/bioelectricity/InfiniteMorphospaceWidget.tsx`](./InfiniteMorphospaceWidget.tsx)
- **5-Page Visual Guide PDF (13-Year-Old Level, Large-Font Separate Diagrams):**  
  `https://fractastical.github.io/morpholib/thompson_1917_visual_guide_for_everyone.pdf`
- **20-Page Full Academic Manuscript PDF:**  
  `https://fractastical.github.io/morpholib/thompson_1917_fish_transformations_manuscript.pdf`

---

## 3 Ways to Integrate Into `bioelectricity.net`

### 1. Zero-Dependency `<iframe />` Embed (30 Seconds)
Paste this into any page on `bioelectricity.net`:

```html
<iframe
  src="https://fractastical.github.io/morpholib/bioelectricity/"
  title="How Nature Builds 10,000 Fishes — Interactive D'Arcy Thompson & Bioelectric Morphospace Guide by Joel Dietz"
  style="width: 100%; height: 920px; border: 1px solid #263859; border-radius: 16px; background: #070b14;"
  loading="lazy"
  allowfullscreen
></iframe>
```

### 2. Native React / Vite Component (`InfiniteMorphospaceWidget.tsx`)
Copy [`InfiniteMorphospaceWidget.tsx`](./InfiniteMorphospaceWidget.tsx) into your `bioelectricity.net` `src/components/` folder and render:

```tsx
import { InfiniteMorphospaceWidget } from './components/InfiniteMorphospaceWidget';

export function MorphospacePage() {
  return <InfiniteMorphospaceWidget />;
}
```

### 3. JSON Feed Ingestion (`bioelectricity_nexus_feed.json`)
Fetch `https://fractastical.github.io/morpholib/bioelectricity/bioelectricity_nexus_feed.json` to automatically populate cards in `/publications`, `/tools`, and `/research`.
