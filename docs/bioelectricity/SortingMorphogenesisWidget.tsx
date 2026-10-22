import React, { useState } from 'react';

const BASE_URL = 'https://fractastical.github.io/morpholib';

export interface SortingMorphogenesisWidgetProps {
  defaultTab?: 'guide' | 'magazine' | 'simulator';
  height?: number;
}

/**
 * Drop-in React component for bioelectricity.net (Bioelectricity Nexus).
 * Module: Self-Sorting Cells, "Platonic Space," and Causal Emergence 2.0
 * Primary Author: Joel Dietz (Project Delphi)
 */
export const SortingMorphogenesisWidget: React.FC<SortingMorphogenesisWidgetProps> = ({
  defaultTab = 'guide',
  height = 920,
}) => {
  const [view, setView] = useState<'guide' | 'magazine' | 'simulator'>(defaultTab);

  return (
    <section
      style={{
        background: '#070b14',
        color: '#f8fafc',
        borderRadius: 16,
        border: '1px solid #263859',
        padding: 28,
        fontFamily: "'Inter', sans-serif",
      }}
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', gap: 12, marginBottom: 16 }}>
        <div>
          <span
            style={{
              background: 'rgba(168, 85, 247, 0.16)',
              border: '1px solid rgba(168, 85, 247, 0.45)',
              color: '#c084fc',
              padding: '4px 12px',
              borderRadius: 999,
              fontSize: 12,
              fontWeight: 700,
            }}
          >
            SORTING MORPHOGENESIS &amp; CAUSAL EMERGENCE 2.0 · BY JOEL DIETZ
          </span>
          <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 25, fontWeight: 700, marginTop: 10 }}>
            Self-Sorting Cells, "Platonic Space," and Why a 6-Line Algorithm Sparked a Huge Debate
          </h2>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
          <button
            onClick={() => setView('guide')}
            style={{
              background: view === 'guide' ? '#38bdf8' : '#131e36',
              color: view === 'guide' ? '#06101e' : '#e2e8f0',
              border: '1px solid #263859',
              borderRadius: 8,
              padding: '8px 14px',
              fontWeight: 700,
              cursor: 'pointer',
            }}
          >
            Illustrated Guide
          </button>
          <button
            onClick={() => setView('magazine')}
            style={{
              background: view === 'magazine' ? '#38bdf8' : '#131e36',
              color: view === 'magazine' ? '#06101e' : '#e2e8f0',
              border: '1px solid #263859',
              borderRadius: 8,
              padding: '8px 14px',
              fontWeight: 700,
              cursor: 'pointer',
            }}
          >
            4-Page Magazine Spread
          </button>
          <button
            onClick={() => setView('simulator')}
            style={{
              background: view === 'simulator' ? '#38bdf8' : '#131e36',
              color: view === 'simulator' ? '#06101e' : '#e2e8f0',
              border: '1px solid #263859',
              borderRadius: 8,
              padding: '8px 14px',
              fontWeight: 700,
              cursor: 'pointer',
            }}
          >
            Interactive Cell Simulator
          </button>
          <a
            href={`${BASE_URL}/zhang_levin_13yo_illustrated_guide.pdf`}
            target="_blank"
            rel="noreferrer"
            style={{
              background: 'rgba(34, 197, 94, 0.15)',
              border: '1px solid rgba(34, 197, 94, 0.45)',
              color: '#4ade80',
              borderRadius: 8,
              padding: '8px 14px',
              fontWeight: 700,
              textDecoration: 'none',
              fontSize: 14,
            }}
          >
            Download 4-Page Magazine PDF
          </a>
        </div>
      </div>

      {view === 'simulator' && (
        <iframe
          src={`${BASE_URL}/bioelectricity/sorting-morphogenesis.html`}
          title="Interactive Self-Sorting Cells & Causal Emergence 2.0 Lab by Joel Dietz"
          style={{ width: '100%', height, border: '1px solid #263859', borderRadius: 12, background: '#070b14' }}
          loading="lazy"
        />
      )}

      {view === 'guide' && (
        <div style={{ display: 'grid', gap: 20 }}>
          <img
            src={`${BASE_URL}/media/zhang_13yo_fig1_three_personalities.png`}
            alt="Illustrated Guide 1: The Three Cell Personalities"
            style={{ width: '100%', borderRadius: 12, border: '1px solid #334155' }}
          />
          <img
            src={`${BASE_URL}/media/zhang_13yo_fig2_two_rulers_illusion.png`}
            alt="Illustrated Guide 2: Why Delayed Gratification Was an Optical Illusion of the Ruler"
            style={{ width: '100%', borderRadius: 12, border: '1px solid #334155' }}
          />
          <img
            src={`${BASE_URL}/media/zhang_13yo_fig3_sports_cars_vs_pedestrians.png`}
            alt="Illustrated Guide 3: Why Secret Cliques Form & C=2 Anti-Clustering"
            style={{ width: '100%', borderRadius: 12, border: '1px solid #334155' }}
          />
          <img
            src={`${BASE_URL}/media/zhang_13yo_fig4_real_data_curves.png`}
            alt="Illustrated Guide 4: Real Computer Experiment Curves"
            style={{ width: '100%', borderRadius: 12, border: '1px solid #334155' }}
          />
        </div>
      )}

      {view === 'magazine' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', gap: 18 }}>
          <img
            src={`${BASE_URL}/media/zhang_13yo_page_1.png`}
            alt="Page 1 of Illustrated Student's Guide PDF"
            style={{ width: '100%', borderRadius: 12, border: '1px solid #334155' }}
          />
          <img
            src={`${BASE_URL}/media/zhang_13yo_page_2.png`}
            alt="Page 2 of Illustrated Student's Guide PDF"
            style={{ width: '100%', borderRadius: 12, border: '1px solid #334155' }}
          />
          <img
            src={`${BASE_URL}/media/zhang_13yo_page_3.png`}
            alt="Page 3 of Illustrated Student's Guide PDF"
            style={{ width: '100%', borderRadius: 12, border: '1px solid #334155' }}
          />
          <img
            src={`${BASE_URL}/media/zhang_13yo_page_4.png`}
            alt="Page 4 of Illustrated Student's Guide PDF"
            style={{ width: '100%', borderRadius: 12, border: '1px solid #334155' }}
          />
        </div>
      )}
    </section>
  );
};

export default SortingMorphogenesisWidget;
