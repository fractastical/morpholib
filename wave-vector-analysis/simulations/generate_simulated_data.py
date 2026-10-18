#!/usr/bin/env python3
"""
Generate simulated Ca²⁺ wave data for testing hypotheses.

This script creates synthetic spark tracking data that mimics real experimental
conditions with configurable:
- Number of embryos
- Poke locations
- Wave propagation patterns
- Timing and dynamics
- Spatial distributions

Output format matches spark_tracks.csv for direct comparison.
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import math


@dataclass
class Embryo:
    """Configuration for a simulated embryo."""
    id: str
    center_x: float
    center_y: float
    length: float  # Length of embryo (pixels)
    width: float   # Width of embryo (pixels)
    angle: float   # Orientation angle (degrees, 0 = horizontal)
    head_x: float  # Head position
    head_y: float
    tail_x: float  # Tail position
    tail_y: float


@dataclass
class PokeConfig:
    """Configuration for a poke event."""
    x: float
    y: float
    embryo_id: Optional[str] = None  # Which embryo was poked
    time: float = 0.0  # Time of poke (relative to experiment start)


@dataclass
class WaveConfig:
    """Configuration for wave propagation."""
    speed_px_per_s: float = 5.0  # Propagation speed
    duration_s: float = 10.0  # How long waves persist
    decay_rate: float = 0.1  # Activity decay over time
    radial: bool = True  # Radial vs directional propagation
    direction_deg: Optional[float] = None  # Direction if not radial
    coupling_range_lengths: float = 2.0  # Inter-embryo signaling distance in embryo lengths (0 = isolated)


class SimulationGenerator:
    """Generate simulated Ca²⁺ wave data."""
    
    def __init__(self, fps=1.0, image_width=2048, image_height=2048):
        self.fps = fps
        self.width = image_width
        self.height = image_height
        self.embryos: List[Embryo] = []
        self.pokes: List[PokeConfig] = []
        self.wave_config = WaveConfig()
        
    def add_embryo(self, embryo: Embryo):
        """Add an embryo to the simulation."""
        self.embryos.append(embryo)
        
    def add_poke(self, poke: PokeConfig):
        """Add a poke event."""
        self.pokes.append(poke)
        
    def set_wave_config(self, config: WaveConfig):
        """Set wave propagation parameters."""
        self.wave_config = config
        
    def _point_in_embryo(self, x: float, y: float, embryo: Embryo) -> bool:
        """Check if a point is within an embryo boundary."""
        # Transform to embryo-relative coordinates
        dx = x - embryo.center_x
        dy = y - embryo.center_y
        
        # Rotate to align with embryo axis
        angle_rad = math.radians(embryo.angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rx = dx * cos_a + dy * sin_a
        ry = -dx * sin_a + dy * cos_a
        
        # Check if within ellipse (approximating embryo shape)
        a = embryo.length / 2
        b = embryo.width / 2
        return (rx / a) ** 2 + (ry / b) ** 2 <= 1.0
        
    def _calculate_ap_position(self, x: float, y: float, embryo: Embryo) -> float:
        """Calculate AP position (0=head, 1=tail) for a point in an embryo."""
        # Vector from head to point
        dx = x - embryo.head_x
        dy = y - embryo.head_y
        
        # Vector along embryo axis
        axis_dx = embryo.tail_x - embryo.head_x
        axis_dy = embryo.tail_y - embryo.head_y
        axis_len = math.hypot(axis_dx, axis_dy)
        
        if axis_len < 1e-6:
            return 0.5  # Default to middle
        
        # Project point onto axis
        proj = (dx * axis_dx + dy * axis_dy) / axis_len
        ap_norm = max(0.0, min(1.0, proj / axis_len))
        
        return ap_norm
        
    def _generate_wave_from_poke(self, poke: PokeConfig, max_time_s: float, 
                                 track_id_start: int, base_seed: Optional[int] = None) -> pd.DataFrame:
        """Generate spark tracks from a single poke event."""
        tracks = []
        track_id = track_id_start
        
        # Determine which embryos are affected
        affected_embryos = []
        poke_embryo = None
        
        if poke.embryo_id:
            # Find the poked embryo
            for emb in self.embryos:
                if emb.id == poke.embryo_id:
                    poke_embryo = emb
                    affected_embryos.append(emb)
                    break
        
        # Check if poke affects other embryos (inter-embryo signaling)
        coupling_mult = getattr(self.wave_config, "coupling_range_lengths", 2.0)
        for emb in self.embryos:
            if emb == poke_embryo:
                continue
            # Check distance between poke and embryo center
            dist = math.hypot(poke.x - emb.center_x, poke.y - emb.center_y)
            # If within coupling range, trigger wave
            if dist < emb.length * coupling_mult:
                affected_embryos.append(emb)
        
        # Generate waves for each affected embryo
        for emb_idx, embryo in enumerate(affected_embryos):
            # Initial spark at poke location (if in embryo) or near embryo surface
            if embryo == poke_embryo and self._point_in_embryo(poke.x, poke.y, embryo):
                start_x, start_y = poke.x, poke.y
            else:
                # Start near embryo center or at closest point to poke
                start_x = embryo.center_x
                start_y = embryo.center_y
            
            # Sub-generator per (poke, embryo) so primary embryo A gets identical RNG stream
            # whether or not secondary embryo B is recruited (paired A/B lock)
            emb_rng = (
                np.random.default_rng(base_seed * 1009 + int(poke.time * 100) + ord(embryo.id[0]) * 37)
                if base_seed is not None
                else np.random.default_rng()
            )
            # Generate multiple spark clusters propagating from start
            n_clusters = int(emb_rng.poisson(3)) + 1  # 1-6 clusters typically
            
            for cluster_idx in range(n_clusters):
                cluster_rng = (
                    np.random.default_rng(base_seed * 10007 + ord(embryo.id[0]) * 131 + cluster_idx * 17)
                    if base_seed is not None
                    else emb_rng
                )
                cluster_tracks = self._generate_cluster_track(
                    start_x, start_y, embryo, poke.time, max_time_s, track_id, rng=cluster_rng
                )
                tracks.extend(cluster_tracks)
                track_id += 1
        
        return pd.DataFrame(tracks)
    
    def _generate_cluster_track(self, start_x: float, start_y: float, 
                                embryo: Embryo, poke_time: float, 
                                max_time_s: float, track_id: int,
                                rng: Optional[np.random.Generator] = None) -> List[dict]:
        """Generate a single cluster track propagating from a starting point."""
        if rng is None:
            rng = np.random.default_rng()
        track = []
        current_x = start_x
        current_y = start_y
        
        # Determine propagation direction
        if self.wave_config.radial:
            # Radial: random direction from poke point
            angle = float(rng.uniform(0, 2 * math.pi))
        else:
            # Directional: use specified direction or random
            if self.wave_config.direction_deg is not None:
                angle = math.radians(self.wave_config.direction_deg)
            else:
                angle = float(rng.uniform(0, 2 * math.pi))
        
        speed = float(rng.normal(self.wave_config.speed_px_per_s, 
                                 self.wave_config.speed_px_per_s * 0.2))
        speed = max(0.5, speed)  # Minimum speed
        
        # Duration of this track
        duration = float(rng.exponential(self.wave_config.duration_s * 0.5))
        duration = min(duration, max_time_s - poke_time)
        
        dt = 1.0 / self.fps
        time = poke_time
        frame_idx = int(poke_time * self.fps)
        
        prev_x, prev_y = current_x, current_y
        
        while time < poke_time + duration and time < max_time_s:
            # Calculate position with some randomness
            dx = math.cos(angle) * speed * dt
            dy = math.sin(angle) * speed * dt
            
            # Add some directional variation (not perfectly straight)
            angle_variation = float(rng.normal(0, 0.1))
            angle += angle_variation
            
            current_x += dx
            current_y += dy
            
            # Bounce back if outside embryo (simplified boundary)
            if not self._point_in_embryo(current_x, current_y, embryo):
                # Reflect direction
                angle += math.pi / 2
                # Move back inside
                current_x = prev_x
                current_y = prev_y
                continue
            
            # Calculate speed for this frame
            vx = (current_x - prev_x) / dt if dt > 0 else 0
            vy = (current_y - prev_y) / dt if dt > 0 else 0
            speed_frame = math.hypot(vx, vy)
            angle_deg = math.degrees(math.atan2(-vy, vx))
            
            # Activity decays over time
            time_since_poke = time - poke_time
            activity_factor = math.exp(-self.wave_config.decay_rate * time_since_poke)
            area = max(10, int(100 * activity_factor * float(rng.uniform(0.5, 1.5))))
            
            # Calculate AP position
            ap_norm = self._calculate_ap_position(current_x, current_y, embryo)
            
            # Distance from poke
            dist_from_poke = math.hypot(current_x - self.pokes[0].x, 
                                       current_y - self.pokes[0].y)
            
            track.append({
                'track_id': track_id,
                'frame_idx': frame_idx,
                'time_s': time - poke_time,  # Relative to poke
                'x': current_x,
                'y': current_y,
                'vx': vx,
                'vy': vy,
                'speed': speed_frame,
                'angle_deg': angle_deg,
                'area': area,
                'embryo_id': embryo.id,
                'ap_norm': ap_norm,
                'dv_px': 0.0,  # Could calculate if needed
                'dist_from_poke_px': dist_from_poke,
                'filename': f'simulated_{embryo.id}.tif'
            })
            
            prev_x, prev_y = current_x, current_y
            time += dt
            frame_idx += 1
        
        return track
    
    def generate(self, duration_s: float = 30.0, output_path: Optional[str] = "simulated_spark_tracks.csv",
                 seed: Optional[int] = None) -> pd.DataFrame:
        """Generate complete simulated dataset."""
        all_tracks = []
        track_id = 0
        
        # Sort pokes by time
        sorted_pokes = sorted(self.pokes, key=lambda p: p.time)
        
        for poke in sorted_pokes:
            poke_tracks = self._generate_wave_from_poke(poke, duration_s, track_id, base_seed=seed)
            if len(poke_tracks) > 0:
                all_tracks.append(poke_tracks)
                track_id = int(poke_tracks['track_id'].max()) + 1
        
        if len(all_tracks) == 0:
            raise ValueError("No tracks generated. Check poke locations and embryo configurations.")
        
        df = pd.concat(all_tracks, ignore_index=True)
        df = df.sort_values(['frame_idx', 'track_id'])
        
        # Save to CSV if output_path is provided
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"✓ Generated {len(df)} track states from {track_id} unique tracks")
            print(f"✓ Saved to {output_path}")
        
        return df


def apply_telemetry_perturbations(
    df: pd.DataFrame,
    dropout_rate: float = 0.0,
    position_jitter_px: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply deterministic synthetic telemetry perturbations (frame dropout and
    sub-pixel localization jitter) modeled on MesmerPrism/rusty-lsl's
    `crates/rusty-lsl-synthetic` harness, while preserving at least 2 states per
    track and recomputing frame-to-frame kinematics (`vx`, `vy`, `speed`, `angle`)."""
    out = df.copy().reset_index(drop=True)
    rng = np.random.default_rng(int(seed) + 9001)

    if position_jitter_px > 0.0 and len(out) > 0:
        jx = rng.normal(0.0, float(position_jitter_px), size=len(out))
        jy = rng.normal(0.0, float(position_jitter_px), size=len(out))
        out["x"] = out["x"].astype(float) + jx
        out["y"] = out["y"].astype(float) + jy

    if 0.0 < dropout_rate < 1.0 and len(out) > 0:
        keep_mask = np.ones(len(out), dtype=bool)
        for tid, group in out.groupby("track_id"):
            idxs = group.index.to_numpy()
            if len(idxs) <= 2:
                continue
            # Never drop the first or last frame of a track so lifetime stays bounded
            inner = idxs[1:-1]
            drop_draw = rng.random(len(inner)) < float(dropout_rate)
            keep_mask[inner[drop_draw]] = False
        out = out[keep_mask].copy().reset_index(drop=True)

    # Recompute per-track frame-to-frame velocities, speeds, and angles if perturbed
    if (dropout_rate > 0.0 or position_jitter_px > 0.0) and len(out) > 0:
        out = out.sort_values(["track_id", "frame_idx"]).reset_index(drop=True)
        for tid, group in out.groupby("track_id"):
            idxs = group.index.to_numpy()
            xs = group["x"].to_numpy(dtype=float)
            ys = group["y"].to_numpy(dtype=float)
            ts = group["time_s"].to_numpy(dtype=float)
            vxs = np.full(len(idxs), np.nan)
            vys = np.full(len(idxs), np.nan)
            spds = np.full(len(idxs), np.nan)
            angs = np.full(len(idxs), np.nan)
            for k in range(1, len(idxs)):
                dt = max(ts[k] - ts[k - 1], 1e-6)
                dx = xs[k] - xs[k - 1]
                dy = ys[k] - ys[k - 1]
                vxs[k] = dx / dt
                vys[k] = dy / dt
                spds[k] = math.hypot(vxs[k], vys[k])
                angs[k] = math.degrees(math.atan2(-dy, dx))
            out.loc[idxs, "vx"] = vxs
            out.loc[idxs, "vy"] = vys
            out.loc[idxs, "speed"] = spds
            out.loc[idxs, "angle"] = angs
        out = out.sort_values(["frame_idx", "track_id"]).reset_index(drop=True)

    return out


def run_telemetry_stress_benchmark(seed: int = 42) -> bool:
    """Self-test / benchmark verifying that `spark_tracks_to_clusters.py` and
    `analyze_oscillatory_coherence()` remain stable under realistic frame dropout
    (5-15%) and sub-pixel localization jitter (0.5-1.5 px)."""
    import sys
    import tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from spark_tracks_to_clusters import process_tracks_to_clusters
    from analyze_experimental_hypotheses import analyze_oscillatory_coherence

    scenarios = create_experiment_scenarios()
    scen = scenarios["two_embryos_coupled"]
    gen = SimulationGenerator(fps=1.0)
    for emb in scen["embryos"]:
        gen.add_embryo(emb)
    for pk in scen["pokes"]:
        gen.add_poke(pk)
    gen.set_wave_config(scen["wave_config"])

    df_base = gen.generate(duration_s=30.0, output_path=None, seed=seed)
    coh_base = analyze_oscillatory_coherence(df_base)

    print("=== Telemetry Perturbation Stress Benchmark (rusty-lsl / PolarH10) ===")
    print(f"  Baseline (dropout=0%, jitter=0.0px): states={len(df_base)}, tracks={df_base['track_id'].nunique()}, "
          f"PLV={coh_base['inter_embryo_plv']:.4f}, H_norm={coh_base['overall_shannon_entropy_norm']:.4f}")

    all_ok = True
    for drop, jit in [(0.05, 0.5), (0.10, 1.0), (0.15, 1.5)]:
        df_pert = apply_telemetry_perturbations(df_base, dropout_rate=drop, position_jitter_px=jit, seed=seed)
        with tempfile.TemporaryDirectory() as tmpd:
            t_csv = Path(tmpd) / "pert_tracks.csv"
            c_csv = Path(tmpd) / "pert_clusters.csv"
            df_pert.to_csv(t_csv, index=False)
            df_clust = process_tracks_to_clusters(str(t_csv), str(c_csv))
        coh_pert = analyze_oscillatory_coherence(df_pert, df_clust)
        tracks_preserved = df_pert["track_id"].nunique() == df_base["track_id"].nunique()
        plv_diff = abs(coh_pert["inter_embryo_plv"] - coh_base["inter_embryo_plv"])
        h_diff = abs(coh_pert["overall_shannon_entropy_norm"] - coh_base["overall_shannon_entropy_norm"])
        ok = tracks_preserved and plv_diff < 0.25 and h_diff < 0.15 and len(df_clust) == df_base["track_id"].nunique()
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] dropout={drop*100:4.1f}%, jitter={jit:.1f}px -> states={len(df_pert):3d}, "
              f"clusters={len(df_clust):2d}, PLV={coh_pert['inter_embryo_plv']:.4f} (|Δ|={plv_diff:.4f}), "
              f"H_norm={coh_pert['overall_shannon_entropy_norm']:.4f} (|Δ|={h_diff:.4f})")
    return all_ok


def create_experiment_scenarios():
    """Create predefined experimental scenarios for testing."""
    scenarios = {}
    
    # Scenario 1: Two embryos, head-head orientation, poke in embryo A
    scenarios['two_embryos_head_head'] = {
        'embryos': [
            Embryo(id='A', center_x=500, center_y=1024, length=400, width=150, 
                  angle=0, head_x=300, head_y=1024, tail_x=700, tail_y=1024),
            Embryo(id='B', center_x=1500, center_y=1024, length=400, width=150,
                  angle=0, head_x=1300, head_y=1024, tail_x=1700, tail_y=1024)
        ],
        'pokes': [PokeConfig(x=500, y=1024, embryo_id='A', time=0.0)],
        'wave_config': WaveConfig(speed_px_per_s=5.0, duration_s=10.0, radial=True)
    }
    
    # Scenario 2: Three embryos in triangle
    scenarios['three_embryos_triangle'] = {
        'embryos': [
            Embryo(id='A', center_x=600, center_y=800, length=300, width=120,
                  angle=0, head_x=450, head_y=800, tail_x=750, tail_y=800),
            Embryo(id='B', center_x=1500, center_y=600, length=300, width=120,
                  angle=45, head_x=1350, head_y=550, tail_x=1650, tail_y=650),
            Embryo(id='C', center_x=1500, center_y=1400, length=300, width=120,
                  angle=-45, head_x=1350, head_y=1450, tail_x=1650, tail_y=1350)
        ],
        'pokes': [PokeConfig(x=600, y=800, embryo_id='A', time=0.0)],
        'wave_config': WaveConfig(speed_px_per_s=6.0, duration_s=12.0, radial=True)
    }
    
    # Scenario 3: Two embryos, tail-tail orientation
    scenarios['two_embryos_tail_tail'] = {
        'embryos': [
            Embryo(id='A', center_x=500, center_y=1024, length=400, width=150,
                  angle=0, head_x=300, head_y=1024, tail_x=700, tail_y=1024),
            Embryo(id='B', center_x=1500, center_y=1024, length=400, width=150,
                  angle=180, head_x=1700, head_y=1024, tail_x=1300, tail_y=1024)
        ],
        'pokes': [PokeConfig(x=700, y=1024, embryo_id='A', time=0.0)],  # Poke at tail
        'wave_config': WaveConfig(speed_px_per_s=5.0, duration_s=10.0, radial=True)
    }
    
    # Scenario 4: Multiple poke locations in same embryo
    scenarios['multiple_pokes_same_embryo'] = {
        'embryos': [
            Embryo(id='A', center_x=1024, center_y=1024, length=500, width=200,
                  angle=0, head_x=774, head_y=1024, tail_x=1274, tail_y=1024)
        ],
        'pokes': [
            PokeConfig(x=900, y=1024, embryo_id='A', time=0.0),    # Anterior
            PokeConfig(x=1100, y=1024, embryo_id='A', time=15.0),  # Posterior, later
        ],
        'wave_config': WaveConfig(speed_px_per_s=4.0, duration_s=8.0, radial=True)
    }
    
    # Scenario 5: Three embryos in a line
    scenarios['three_embryos_line'] = {
        'embryos': [
            Embryo(id='A', center_x=400, center_y=1024, length=350, width=140,
                  angle=0, head_x=225, head_y=1024, tail_x=575, tail_y=1024),
            Embryo(id='B', center_x=1024, center_y=1024, length=350, width=140,
                  angle=0, head_x=849, head_y=1024, tail_x=1199, tail_y=1024),
            Embryo(id='C', center_x=1648, center_y=1024, length=350, width=140,
                  angle=0, head_x=1473, head_y=1024, tail_x=1823, tail_y=1024)
        ],
        'pokes': [PokeConfig(x=400, y=1024, embryo_id='A', time=0.0)],
        'wave_config': WaveConfig(speed_px_per_s=5.5, duration_s=12.0, radial=True)
    }
    
    # Scenario 6: Four embryos in a square formation
    scenarios['four_embryos_square'] = {
        'embryos': [
            Embryo(id='A', center_x=600, center_y=600, length=300, width=120,
                  angle=0, head_x=450, head_y=600, tail_x=750, tail_y=600),
            Embryo(id='B', center_x=1444, center_y=600, length=300, width=120,
                  angle=0, head_x=1294, head_y=600, tail_x=1594, tail_y=600),
            Embryo(id='C', center_x=600, center_y=1448, length=300, width=120,
                  angle=0, head_x=450, head_y=1448, tail_x=750, tail_y=1448),
            Embryo(id='D', center_x=1444, center_y=1448, length=300, width=120,
                  angle=0, head_x=1294, head_y=1448, tail_x=1594, tail_y=1448)
        ],
        'pokes': [PokeConfig(x=600, y=600, embryo_id='A', time=0.0)],
        'wave_config': WaveConfig(speed_px_per_s=6.0, duration_s=15.0, radial=True)
    }
    
    # Scenario 7: Four embryos in a line
    scenarios['four_embryos_line'] = {
        'embryos': [
            Embryo(id='A', center_x=320, center_y=1024, length=300, width=120,
                  angle=0, head_x=170, head_y=1024, tail_x=470, tail_y=1024),
            Embryo(id='B', center_x=682, center_y=1024, length=300, width=120,
                  angle=0, head_x=532, head_y=1024, tail_x=832, tail_y=1024),
            Embryo(id='C', center_x=1044, center_y=1024, length=300, width=120,
                  angle=0, head_x=894, head_y=1024, tail_x=1194, tail_y=1024),
            Embryo(id='D', center_x=1406, center_y=1024, length=300, width=120,
                  angle=0, head_x=1256, head_y=1024, tail_x=1556, tail_y=1024)
        ],
        'pokes': [PokeConfig(x=320, y=1024, embryo_id='A', time=0.0)],
        'wave_config': WaveConfig(speed_px_per_s=5.5, duration_s=15.0, radial=True)
    }
    
    # Scenario 8: Three embryos, central one poked
    scenarios['three_embryos_central_poke'] = {
        'embryos': [
            Embryo(id='A', center_x=500, center_y=1024, length=350, width=140,
                  angle=0, head_x=325, head_y=1024, tail_x=675, tail_y=1024),
            Embryo(id='B', center_x=1024, center_y=1024, length=350, width=140,
                  angle=0, head_x=849, head_y=1024, tail_x=1199, tail_y=1024),
            Embryo(id='C', center_x=1548, center_y=1024, length=350, width=140,
                  angle=0, head_x=1373, head_y=1024, tail_x=1723, tail_y=1024)
        ],
        'pokes': [PokeConfig(x=1024, y=1024, embryo_id='B', time=0.0)],  # Poke middle embryo
        'wave_config': WaveConfig(speed_px_per_s=5.5, duration_s=12.0, radial=True, coupling_range_lengths=2.0)
    }

    # Scenario 9: Two embryos head-head, inter-embryo coupling enabled (center distance 700px < 2.0 * 400px)
    scenarios['two_embryos_coupled'] = {
        'embryos': [
            Embryo(id='A', center_x=650, center_y=1024, length=400, width=150,
                  angle=0, head_x=450, head_y=1024, tail_x=850, tail_y=1024),
            Embryo(id='B', center_x=1350, center_y=1024, length=400, width=150,
                  angle=0, head_x=1150, head_y=1024, tail_x=1550, tail_y=1024)
        ],
        'pokes': [PokeConfig(x=650, y=1024, embryo_id='A', time=0.0)],
        'wave_config': WaveConfig(speed_px_per_s=5.0, duration_s=10.0, radial=True, coupling_range_lengths=2.0)
    }

    # Scenario 10: Two embryos head-head, inter-embryo coupling blocked (negative control, coupling_range_lengths=0.0)
    scenarios['two_embryos_isolated_control'] = {
        'embryos': [
            Embryo(id='A', center_x=650, center_y=1024, length=400, width=150,
                  angle=0, head_x=450, head_y=1024, tail_x=850, tail_y=1024),
            Embryo(id='B', center_x=1350, center_y=1024, length=400, width=150,
                  angle=0, head_x=1150, head_y=1024, tail_x=1550, tail_y=1024)
        ],
        'pokes': [PokeConfig(x=650, y=1024, embryo_id='A', time=0.0)],
        'wave_config': WaveConfig(speed_px_per_s=5.0, duration_s=10.0, radial=True, coupling_range_lengths=0.0)
    }
    
    return scenarios


def validate_wave_catalog(catalog_path: Optional[Path] = None) -> Tuple[bool, List[str]]:
    """Validate `wave-vector-analysis/simulations/catalog.json` 4-tier claim-boundary schema."""
    path = catalog_path or (Path(__file__).resolve().parent / "catalog.json")
    errors: List[str] = []
    if not path.exists():
        return False, [f"Catalog file not found: {path}"]
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "combinatorial-swarmability-catalog.v1":
        errors.append(f"Unexpected schema_version: {data.get('schema_version')}")
    for idx, entry in enumerate(data.get("entries", [])):
        eid = entry.get("id", f"entry_{idx}")
        cb = entry.get("claim_boundary", {})
        for tier in ("reported", "reconstruction", "transfer_boundary", "does_not_claim"):
            items = cb.get(tier)
            if not isinstance(items, list) or len(items) == 0:
                errors.append(f"Entry '{eid}' missing non-empty claim_boundary.{tier}")
    return len(errors) == 0, errors


def generate_paired_ab(
    scenario_a: str,
    scenario_b: str,
    duration_s: float = 30.0,
    fps: float = 1.0,
    seed: int = 42,
    output_prefix: str = "sim_paired_ab",
) -> dict:
    """
    Run Scenario A (Lane A) and Scenario B (Lane B) from the exact same RNG seed,
    writing `<output_prefix>_lane_a.csv`, `<output_prefix>_lane_b.csv`, and
    `<output_prefix>_summary.json`.
    """
    scenarios = create_experiment_scenarios()
    if scenario_a not in scenarios or scenario_b not in scenarios:
        raise ValueError(f"Unknown scenario in ({scenario_a}, {scenario_b}). Available: {list(scenarios.keys())}")

    def _run_one(scen_key: str, out_csv: str) -> Tuple[pd.DataFrame, SimulationGenerator]:
        gen = SimulationGenerator(fps=fps)
        sc = scenarios[scen_key]
        for emb in sc["embryos"]:
            gen.add_embryo(emb)
        for poke in sc["pokes"]:
            gen.add_poke(poke)
        gen.set_wave_config(sc["wave_config"])
        df = gen.generate(duration_s=duration_s, output_path=out_csv, seed=seed)
        meta_path = out_csv.replace(".csv", "_metadata.json")
        with open(meta_path, "w") as fh:
            json.dump({
                "scenario": scen_key,
                "seed": seed,
                "duration_s": duration_s,
                "fps": fps,
                "embryos": [asdict(e) for e in gen.embryos],
                "pokes": [asdict(p) for p in gen.pokes],
                "wave_config": asdict(gen.wave_config),
            }, fh, indent=2)
        return df, gen

    csv_a = f"{output_prefix}_lane_a.csv"
    csv_b = f"{output_prefix}_lane_b.csv"
    df_a, _ = _run_one(scenario_a, csv_a)
    df_b, _ = _run_one(scenario_b, csv_b)

    # Check primary-embryo A trajectory identity when primary embryo & wave kinetics match
    df_a_embA = df_a[df_a["embryo_id"] == "A"].reset_index(drop=True)
    df_b_embA = df_b[df_b["embryo_id"] == "A"].reset_index(drop=True)
    primary_identical = (
        len(df_a_embA) == len(df_b_embA)
        and np.allclose(df_a_embA["x"].values, df_b_embA["x"].values)
        and np.allclose(df_a_embA["y"].values, df_b_embA["y"].values)
    )

    summary = {
        "scenario_a": scenario_a,
        "scenario_b": scenario_b,
        "seed": seed,
        "duration_s": duration_s,
        "fps": fps,
        "lane_a_csv": csv_a,
        "lane_b_csv": csv_b,
        "primary_embryo_a_seed_locked_identical": bool(primary_identical),
        "lane_a_total_states": int(len(df_a)),
        "lane_b_total_states": int(len(df_b)),
        "delta_total_states_b_minus_a": int(len(df_b) - len(df_a)),
        "lane_a_unique_tracks": int(df_a["track_id"].nunique()),
        "lane_b_unique_tracks": int(df_b["track_id"].nunique()),
        "delta_unique_tracks_b_minus_a": int(df_b["track_id"].nunique() - df_a["track_id"].nunique()),
        "lane_a_embryos_recruited": sorted(df_a["embryo_id"].dropna().unique().tolist()),
        "lane_b_embryos_recruited": sorted(df_b["embryo_id"].dropna().unique().tolist()),
        "lane_a_total_area": int(df_a["area"].sum()),
        "lane_b_total_area": int(df_b["area"].sum()),
        "delta_total_area_b_minus_a": int(df_b["area"].sum() - df_a["area"].sum()),
    }
    summary_path = f"{output_prefix}_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"✓ Saved paired A/B summary to {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate simulated Ca²⁺ wave data for hypothesis testing"
    )
    parser.add_argument(
        '--scenario',
        choices=['two_embryos_head_head', 'three_embryos_triangle', 
                'two_embryos_tail_tail', 'multiple_pokes_same_embryo',
                'three_embryos_line', 'four_embryos_square', 'four_embryos_line',
                'three_embryos_central_poke', 'two_embryos_coupled',
                'two_embryos_isolated_control', 'custom'],
        default='two_embryos_head_head',
        help='Predefined experimental scenario'
    )
    parser.add_argument(
        '--output',
        default='simulated_spark_tracks.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=30.0,
        help='Simulation duration in seconds (default: 30.0)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=1.0,
        help='Frames per second (default: 1.0)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Deterministic RNG seed (default: 42)'
    )
    parser.add_argument(
        '--dropout-rate',
        type=float,
        default=0.0,
        help='Synthetic frame-dropout rate in [0.0, 0.9] (modeled on MesmerPrism/rusty-lsl)'
    )
    parser.add_argument(
        '--position-jitter-px',
        type=float,
        default=0.0,
        help='Gaussian localization jitter standard deviation in pixels (default: 0.0)'
    )
    parser.add_argument(
        '--self-test',
        action='store_true',
        help='Run synthetic telemetry perturbation stress benchmark (5-15%% dropout + jitter)'
    )
    parser.add_argument(
        '--compare-ab',
        nargs=2,
        metavar=('SCENARIO_A', 'SCENARIO_B'),
        default=None,
        help='Run deterministic paired A/B comparison between two scenarios with locked seed'
    )
    parser.add_argument(
        '--config',
        help='JSON config file (from calibrate_from_real_data.py or custom config)'
    )
    
    args = parser.parse_args()

    cat_ok, cat_errs = validate_wave_catalog()
    if not cat_ok:
        raise RuntimeError(f"Wave simulation catalog.json failed validation: {cat_errs}")

    if args.self_test:
        print(f"[PASS] Claim-boundary catalog (wave-vector-analysis/simulations/catalog.json): 0 error(s)")
        ok = run_telemetry_stress_benchmark(seed=args.seed)
        if not ok:
            sys.exit(1)
        return

    if args.compare_ab:
        scen_a, scen_b = args.compare_ab
        prefix = args.output.replace('.csv', '')
        summary = generate_paired_ab(
            scenario_a=scen_a,
            scenario_b=scen_b,
            duration_s=args.duration,
            fps=args.fps,
            seed=args.seed,
            output_prefix=prefix,
        )
        print(f"[PASS] Claim-boundary catalog (wave-vector-analysis/simulations/catalog.json): 0 error(s)")
        print(f"=== Paired Wave Simulation Summary ({scen_a} vs {scen_b}, seed={args.seed}) ===")
        print(f"  Primary Embryo A trajectories seed-locked identical: {summary['primary_embryo_a_seed_locked_identical']}")
        print(f"  Lane A ({scen_a}): {summary['lane_a_total_states']} states, {summary['lane_a_unique_tracks']} tracks, embryos={summary['lane_a_embryos_recruited']}")
        print(f"  Lane B ({scen_b}): {summary['lane_b_total_states']} states, {summary['lane_b_unique_tracks']} tracks, embryos={summary['lane_b_embryos_recruited']}")
        print(f"  Paired Delta (B - A): Δstates={summary['delta_total_states_b_minus_a']:+d}, Δtracks={summary['delta_unique_tracks_b_minus_a']:+d}, Δarea={summary['delta_total_area_b_minus_a']:+d}")
        return
    
    # Create generator
    gen = SimulationGenerator(fps=args.fps)
    
    # Check if config file is provided (takes precedence over scenario)
    scenario_name = args.scenario  # Default scenario name
    
    if args.config and Path(args.config).exists():
        # Load config from JSON file (from real data calibration or custom)
        print(f"Loading configuration from {args.config}...")
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Load embryos
        for emb_dict in config.get('embryos', []):
            embryo = Embryo(**emb_dict)
            gen.add_embryo(embryo)
        
        # Load pokes
        for poke_dict in config.get('pokes', []):
            poke = PokeConfig(**poke_dict)
            gen.add_poke(poke)
        
        # Load wave config
        if 'wave_config' in config:
            wave_dict = config['wave_config']
            wave_config = WaveConfig(**wave_dict)
            gen.set_wave_config(wave_config)
        
        print(f"  → Loaded {len(gen.embryos)} embryo(s), {len(gen.pokes)} poke(s)")
        
        if 'source_data' in config:
            print(f"  → Configuration derived from real data")
            source = config['source_data']
            if 'tracks_csv' in source:
                print(f"     Source tracks: {Path(source['tracks_csv']).name}")
        
        scenario_name = "calibrated from real data"
    elif args.config:
        print(f"Error: Config file '{args.config}' not found.")
        return
    else:
        # Use predefined scenario
        scenarios = create_experiment_scenarios()
        if args.scenario not in scenarios:
            print(f"Error: Unknown scenario '{args.scenario}'")
            print(f"Available scenarios: {list(scenarios.keys())}")
            return
        scenario = scenarios[args.scenario]
        
        # Set up embryos
        for embryo in scenario['embryos']:
            gen.add_embryo(embryo)
        
        # Set up pokes
        for poke in scenario['pokes']:
            gen.add_poke(poke)
        
        # Set wave config
        gen.set_wave_config(scenario['wave_config'])
    
    # Generate simulation
    print(f"Generating simulation: {scenario_name}")
    print(f"  Embryos: {len(gen.embryos)}")
    print(f"  Pokes: {len(gen.pokes)}")
    print(f"  Duration: {args.duration}s at {args.fps} fps")
    print()
    
    df = gen.generate(duration_s=args.duration, output_path=None, seed=args.seed)
    if args.dropout_rate > 0.0 or args.position_jitter_px > 0.0:
        df = apply_telemetry_perturbations(
            df,
            dropout_rate=args.dropout_rate,
            position_jitter_px=args.position_jitter_px,
            seed=args.seed,
        )
        print(f"✓ Applied telemetry perturbations (dropout_rate={args.dropout_rate:.2f}, "
              f"position_jitter_px={args.position_jitter_px:.2f}) -> {len(df)} states")
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"✓ Saved {len(df)} track states to {args.output}")
    
    # Save metadata
    metadata_path = args.output.replace('.csv', '_metadata.json')
    metadata = {
        'scenario': args.scenario,
        'duration_s': args.duration,
        'fps': args.fps,
        'n_embryos': len(gen.embryos),
        'n_pokes': len(gen.pokes),
        'embryos': [asdict(emb) for emb in gen.embryos],
        'pokes': [asdict(poke) for poke in gen.pokes],
        'wave_config': asdict(gen.wave_config),
        'n_tracks': df['track_id'].nunique(),
        'n_track_states': len(df)
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata to {metadata_path}")
    
    # Generate summary statistics
    print("\n=== Simulation Summary ===")
    print(f"Total track states: {len(df)}")
    print(f"Unique tracks: {df['track_id'].nunique()}")
    print(f"Time range: {df['time_s'].min():.2f} to {df['time_s'].max():.2f} seconds")
    print(f"Embryos represented: {df['embryo_id'].dropna().unique()}")
    print(f"Files: {df['filename'].nunique()}")


if __name__ == "__main__":
    main()

