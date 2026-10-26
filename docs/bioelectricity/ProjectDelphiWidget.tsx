import React, { useState } from 'react';

const BASE_URL = 'https://fractastical.github.io/morpholib';

export interface ProjectDelphiWidgetProps {
  defaultTab?: 'guide' | 'sandbox' | 'bioelectric';
  height?: number;
}

/**
 * Drop-in React component for bioelectricity.net (Bioelectricity Nexus).
 * Primary Author: Joel Dietz (Project Delphi)
 */
export const ProjectDelphiWidget: React.FC<ProjectDelphiWidgetProps> = ({
  height = 900,
}) => {
  const [view, setView] = useState<'cards' | 'interactive'>('cards');

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
              background: 'rgba(56, 189, 248, 0.14)',
              border: '1px solid rgba(56, 189, 248, 0.4)',
              color: '#38bdf8',
              padding: '4px 12px',
              borderRadius: 999,
              fontSize: 12,
              fontWeight: 700,
            }}
          >
            RESEARCH MODULE · BY JOEL DIETZ
          </span>
          <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 26, fontWeight: 700, marginTop: 10 }}>
            How Nature Builds 10,000 Different Fishes: Bioelectric Morphospace vs. Trophic Competition
          </h2>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button
            onClick={() => setView('cards')}
            style={{
              background: view === 'cards' ? '#38bdf8' : '#131e36',
              color: view === 'cards' ? '#06101e' : '#e2e8f0',
              border: '1px solid #263859',
              borderRadius: 8,
              padding: '8px 14px',
              fontWeight: 700,
              cursor: 'pointer',
            }}
          >
            Visual Guide
          </button>
          <button
            onClick={() => setView('interactive')}
            style={{
              background: view === 'interactive' ? '#38bdf8' : '#131e36',
              color: view === 'interactive' ? '#06101e' : '#e2e8f0',
              border: '1px solid #263859',
              borderRadius: 8,
              padding: '8px 14px',
              fontWeight: 700,
              cursor: 'pointer',
            }}
          >
            Interactive 600-pt Grid Lab
          </button>
          <a
            href={`${BASE_URL}/thompson_1917_visual_guide_for_everyone.pdf`}
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
            Download 5-Page Visual Guide PDF
          </a>
        </div>
      </div>

      {view === 'interactive' ? (
        <iframe
          src={`${BASE_URL}/bioelectricity/`}
          title="Interactive D'Arcy Thompson & Bioelectric Morphospace Lab by Joel Dietz"
          style={{ width: '100%', height, border: '1px solid #263859', borderRadius: 12, background: '#070b14' }}
          loading="lazy"
        />
      ) : (
        <div style={{ display: 'grid', gap: 20 }}>
          <img
            src={`${BASE_URL}/guide_step1_1917_rubber_sheet.png`}
            alt="Step 1: D'Arcy Thompson's 1917 Rubber Graph-Paper Idea"
            style={{ width: '100%', borderRadius: 12, background: '#fff' }}
          />
          <img
            src={`${BASE_URL}/guide_step2_17dot_head_trap.png`}
            alt="Step 2: The 17-Dot Head Seesaw Trap"
            style={{ width: '100%', borderRadius: 12, background: '#fff' }}
          />
          <img
            src={`${BASE_URL}/guide_step3_box_cheat_vs_curves.png`}
            alt="Step 3: Plain Box Stretch vs Thompson's Curved Grids"
            style={{ width: '100%', borderRadius: 12, background: '#fff' }}
          />
          <img
            src={`${BASE_URL}/guide_step4_outside_vs_inside_rules.png`}
            alt="Step 4: Two Simple Rules of Fish Evolution Across 10,303 Species"
            style={{ width: '100%', borderRadius: 12, background: '#fff' }}
          />
          <img
            src={`${BASE_URL}/guide_step5_10000_fishes_scoreboard.png`}
            alt="Step 5: The Four-Number Scoreboard Across 10,303 Species"
            style={{ width: '100%', borderRadius: 12, background: '#fff' }}
          />
        </div>
      )}
    </section>
  );
};

export default ProjectDelphiWidget;
