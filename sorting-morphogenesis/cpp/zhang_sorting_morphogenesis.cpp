// ============================================================================
// sorting-morphogenesis/cpp/zhang_sorting_morphogenesis.cpp
//
// Standalone C++20 Port of Zhang, Goldstein & Levin (2025):
//   "Classical sorting algorithms as a model of morphogenesis: Self-sorting
//    arrays reveal unexpected competencies in a minimal model of basal
//    intelligence." Adaptive Behavior 33(1):25-54 (2025).
//   DOI: 10.1177/10597123241269740  |  arXiv: 2401.05375
//   Upstream Python repo: https://github.com/Zhangtaining/cell_research
//   Pinned upstream commit: 1fd2bd5921c1f6b423a71f691d5189106a8a1020
//
// Exact Upstream File & Line Traceability:
//   [U1] BubbleSortCell.py (L38-74)       -> bubble_should_move(), propose_bubble()
//   [U2] InsertionSortCell.py (L68-102)   -> left_sorted_with_frozen_reset(), propose_insertion()
//   [U3] SelectionSortCell.py (L41-88) &
//        MultiThreadCell.py (L104-118)    -> propose_selection(), reset_selection_target()
//   [U4] SelectionSortCell.cpython-310.pyc-> selection_movable_pyc310 branch (Paper Fig. 5B)
//   [U5] delay_gratification_analysis_for_not_move.py (L24-89)
//                                         -> discrepancy_runs(), delayed_gratification()
//   [U6] cell_type_aggregation_analysis.py (L518-538)
//                                         -> aggregation_value()
//
// Cross-Language Deterministic Equivalence:
//   Uses an explicit 64-bit SplitMix64 PRNG (`SplitMix64`) shared bit-for-bit
//   with `sorting-morphogenesis/js/zhang_sorting_morphogenesis.mjs` and
//   `sorting-morphogenesis/verify_multilang_ports.py`, guaranteeing 0 cell/step
//   divergence across C++20, JavaScript (Node.js), and Python.
// ============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// Portable 64-bit SplitMix64 PRNG (identical across C++20, Node.js, and Python)
// ----------------------------------------------------------------------------
struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t seed) : state(seed) {}

    uint64_t next_u64() {
        state += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    // Uniform integer in [0, bound - 1] via deterministic modulo
    uint32_t next_uint(uint32_t bound) {
        return static_cast<uint32_t>(next_u64() % static_cast<uint64_t>(bound));
    }

    // Uniform double in [0.0, 1.0) using top 53 bits
    double next_double() {
        return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
    }

    // Deterministic Fisher-Yates shuffle matching JS and Python SplitMix64 ports
    template <typename T>
    void shuffle(std::vector<T>& arr) {
        for (size_t i = arr.size(); i > 1; --i) {
            uint32_t j = next_uint(static_cast<uint32_t>(i));
            std::swap(arr[i - 1], arr[j]);
        }
    }
};

enum class Algotype : uint8_t { Bubble = 0, Insertion = 1, Selection = 2 };
enum class FrozenKind : uint8_t { None = 0, Movable = 1, Immovable = 2 };

struct Cell {
    int value;
    Algotype algotype;
    FrozenKind frozen;
    int target;     // SelectionSortCell.ideal_position pointer
    int direction;  // +1 = increasing, -1 = decreasing
};

// ----------------------------------------------------------------------------
// Metrics: Monotonicity Error, Sortedness, Discrepancy Runs, Delayed
// Gratification ([U5]), and Algotype Aggregation ([U6])
// ----------------------------------------------------------------------------
int monotonicity_error(const std::vector<Cell>& cells, int direction = 1) {
    int err = 0;
    for (size_t i = 1; i < cells.size(); ++i) {
        if (direction == 1) {
            if (cells[i].value < cells[i - 1].value) ++err;
        } else {
            if (cells[i].value > cells[i - 1].value) ++err;
        }
    }
    return err;
}

double sortedness(const std::vector<Cell>& cells, int direction = 1) {
    if (cells.size() <= 1) return 1.0;
    int n = static_cast<int>(cells.size());
    return static_cast<double>(n - 1 - monotonicity_error(cells, direction)) / static_cast<double>(n - 1);
}

// [U6] analysis/cell_type_aggregation_analysis.py (L518-538)
double aggregation_value(const std::vector<Cell>& cells) {
    if (cells.size() <= 1) return 1.0;
    int same = 0;
    for (size_t i = 0; i + 1 < cells.size(); ++i) {
        if (cells[i].algotype == cells[i + 1].algotype) ++same;
    }
    return static_cast<double>(same) / static_cast<double>(cells.size());
}

// [U5] analysis/delay_gratification_analysis_for_not_move.py (L24-89)
double delayed_gratification(const std::vector<int>& mono_trace, bool net = true) {
    std::vector<int> dedup;
    dedup.reserve(mono_trace.size());
    for (int v : mono_trace) {
        if (dedup.empty() || dedup.back() != v) dedup.push_back(v);
    }
    if (dedup.size() < 2) return 0.0;

    std::vector<int> temp(dedup.size() - 1);
    for (size_t i = 1; i < dedup.size(); ++i) {
        temp[i - 1] = dedup[i] - dedup[i - 1];
    }

    std::vector<int> runs;
    int cur = temp[0];
    for (size_t i = 1; i < temp.size(); ++i) {
        if ((temp[i - 1] > 0 && temp[i] > 0) || (temp[i - 1] < 0 && temp[i] < 0)) {
            cur += temp[i];
        } else {
            runs.push_back(cur);
            cur = temp[i];
        }
    }
    runs.push_back(cur);

    size_t first_pos = runs.size();
    for (size_t i = 0; i < runs.size(); ++i) {
        if (runs[i] > 0) {
            first_pos = i;
            break;
        }
    }
    if (first_pos >= runs.size() - 1) return 0.0;

    double total = 0.0;
    int count = 0;
    for (size_t i = first_pos; i + 1 < runs.size(); i += 2) {
        if (runs[i + 1] > 0) break;
        double y = static_cast<double>(runs[i]);
        double x_neg = static_cast<double>(runs[i + 1]);
        double change = net ? (-(x_neg + y) / y) : (-x_neg / y);
        total += change;
        ++count;
    }
    return count > 0 ? (total / static_cast<double>(count)) : 0.0;
}

// ----------------------------------------------------------------------------
// Autonomous Cell-View Perception-Action Rules ([U1], [U2], [U3], [U4])
// ----------------------------------------------------------------------------

// [U2] InsertionSortCell.is_enable_to_move (L68-82):
// Frozen cells reset `prev` so an out-of-order frozen cell does not block all
// Insertion cells to its right.
bool left_sorted_with_frozen_reset(const std::vector<Cell>& cells, int p, int direction) {
    if (direction == 1) {
        int prev = -1000000000;
        for (int k = 0; k < p; ++k) {
            if (cells[k].frozen != FrozenKind::None) {
                prev = -1000000000;
                continue;
            }
            if (cells[k].value < prev) return false;
            prev = cells[k].value;
        }
        return true;
    } else {
        int prev = 1000000000;
        for (int k = 0; k < p; ++k) {
            if (cells[k].frozen != FrozenKind::None) {
                prev = 1000000000;
                continue;
            }
            if (cells[k].value > prev) return false;
            prev = cells[k].value;
        }
        return true;
    }
}

bool cell_should_move(const std::vector<Cell>& cells, int p) {
    const Cell& c = cells[p];
    if (c.frozen != FrozenKind::None) return false;
    int n = static_cast<int>(cells.size());
    int d = c.direction;

    if (c.algotype == Algotype::Bubble) {
        bool left_ok = (p > 0) && (cells[p - 1].frozen != FrozenKind::Immovable) &&
                       ((d == 1) ? (c.value < cells[p - 1].value) : (c.value > cells[p - 1].value));
        bool right_ok = (p < n - 1) && (cells[p + 1].frozen != FrozenKind::Immovable) &&
                        ((d == 1) ? (c.value > cells[p + 1].value) : (c.value < cells[p + 1].value));
        return left_ok || right_ok;
    }
    if (c.algotype == Algotype::Insertion) {
        if (p <= 0 || cells[p - 1].frozen == FrozenKind::Immovable) return false;
        if (!left_sorted_with_frozen_reset(cells, p, d)) return false;
        return (d == 1) ? (c.value < cells[p - 1].value) : (c.value > cells[p - 1].value);
    }
    if (c.algotype == Algotype::Selection) {
        return (p != c.target) && (c.target >= 0) && (c.target < n);
    }
    return false;
}

std::vector<int> compute_active_indices(const std::vector<Cell>& cells) {
    int n = static_cast<int>(cells.size());
    std::vector<int> act;
    act.reserve(n);
    bool inc_ok = true;
    bool dec_ok = true;
    int prev_inc = -1000000000;
    int prev_dec = 1000000000;

    for (int p = 0; p < n; ++p) {
        const Cell& c = cells[p];
        if (c.frozen == FrozenKind::None) {
            int d = c.direction;
            if (c.algotype == Algotype::Bubble) {
                bool left_ok = (p > 0) && (cells[p - 1].frozen != FrozenKind::Immovable) &&
                               ((d == 1) ? (c.value < cells[p - 1].value) : (c.value > cells[p - 1].value));
                bool right_ok = (p < n - 1) && (cells[p + 1].frozen != FrozenKind::Immovable) &&
                                ((d == 1) ? (c.value > cells[p + 1].value) : (c.value < cells[p + 1].value));
                if (left_ok || right_ok) act.push_back(p);
            } else if (c.algotype == Algotype::Insertion) {
                if (p > 0 && cells[p - 1].frozen != FrozenKind::Immovable) {
                    if (d == 1 && inc_ok && c.value < cells[p - 1].value) act.push_back(p);
                    else if (d == -1 && dec_ok && c.value > cells[p - 1].value) act.push_back(p);
                }
            } else if (c.algotype == Algotype::Selection) {
                if (p != c.target && c.target >= 0 && c.target < n) act.push_back(p);
            }
        }
        if (c.frozen != FrozenKind::None) {
            prev_inc = -1000000000;
            prev_dec = 1000000000;
        } else {
            if (c.value < prev_inc) inc_ok = false;
            prev_inc = c.value;
            if (c.value > prev_dec) dec_ok = false;
            prev_dec = c.value;
        }
    }
    return act;
}

struct SimResult {
    int swaps = 0;
    int comparisons = 0;
    int final_error = 0;
    double dg = 0.0;
    double peak_agg = 0.0;
    double final_agg = 0.0;
    double final_sortedness = 0.0;
    int free_inversions = 0;
    std::vector<std::vector<int>> value_snapshots;
    std::vector<int> final_values;
};

SimResult simulate_cell_view(
    std::vector<Cell> cells,
    uint64_t seed,
    bool selection_reset_on_swap = true,
    bool record_snapshots = false,
    int max_swaps = -1,
    int max_steps = -1,
    bool stop_when_sorted = true
) {
    SplitMix64 rng(seed);
    int n = static_cast<int>(cells.size());
    if (max_steps < 0) max_steps = n * n * 40;

    std::vector<int> mono_trace;
    mono_trace.push_back(monotonicity_error(cells));
    double peak_agg = aggregation_value(cells);
    double last_agg = peak_agg;

    SimResult out;
    if (record_snapshots) {
        std::vector<int> row(n);
        for (int i = 0; i < n; ++i) row[i] = cells[i].value;
        out.value_snapshots.push_back(row);
    }

    int swaps = 0;
    int comparisons = 0;

    for (int step = 0; step < max_steps; ++step) {
        if (stop_when_sorted && monotonicity_error(cells) == 0) break;
        if (max_swaps > 0 && swaps >= max_swaps) break;

        std::vector<int> active = compute_active_indices(cells);
        if (active.empty()) break;

        int p = active[rng.next_uint(static_cast<uint32_t>(active.size()))];
        Cell& c = cells[p];
        int d = c.direction;
        int swap_q = -1;
        bool did_compare = false;

        if (c.algotype == Algotype::Bubble) {
            did_compare = true;
            int side = (rng.next_double() < 0.5) ? -1 : 1;
            int q = p + side;
            if (q >= 0 && q < n && cells[q].frozen != FrozenKind::Immovable) {
                if (d == 1) {
                    if ((side == -1 && c.value < cells[q].value) || (side == 1 && c.value > cells[q].value)) {
                        swap_q = q;
                    }
                } else {
                    if ((side == -1 && c.value > cells[q].value) || (side == 1 && c.value < cells[q].value)) {
                        swap_q = q;
                    }
                }
            }
        } else if (c.algotype == Algotype::Insertion) {
            if (p > 0 && left_sorted_with_frozen_reset(cells, p, d) && cells[p - 1].frozen != FrozenKind::Immovable) {
                if ((d == 1 && c.value < cells[p - 1].value) || (d == -1 && c.value > cells[p - 1].value)) {
                    swap_q = p - 1;
                    did_compare = true;
                }
            }
        } else if (c.algotype == Algotype::Selection) {
            int t = c.target;
            if (t >= 0 && t < n && t != p) {
                did_compare = true;
                if (cells[t].frozen == FrozenKind::Immovable) {
                    c.target = (d == 1) ? (t + 1) : (t - 1);
                } else if (c.value < cells[t].value) {
                    swap_q = t;
                } else {
                    c.target = (d == 1) ? (t + 1) : (t - 1);
                }
            }
        }

        if (did_compare) ++comparisons;
        if (swap_q >= 0) {
            std::swap(cells[p], cells[swap_q]);
            if (selection_reset_on_swap) {
                // [U3] MultiThreadCell.swap() calls SelectionSortCell.update()
                if (cells[p].algotype == Algotype::Selection) {
                    cells[p].target = (cells[p].direction == 1) ? 0 : (n - 1);
                }
                if (cells[swap_q].algotype == Algotype::Selection) {
                    cells[swap_q].target = (cells[swap_q].direction == 1) ? 0 : (n - 1);
                }
            } else {
                for (int idx : {p, swap_q}) {
                    if (cells[idx].algotype == Algotype::Selection) {
                        if (cells[idx].direction == 1 && cells[idx].target > idx) {
                            cells[idx].target = 0;
                        } else if (cells[idx].direction == -1 && cells[idx].target < idx) {
                            cells[idx].target = n - 1;
                        }
                    }
                }
            }
            ++swaps;
            mono_trace.push_back(monotonicity_error(cells));
            last_agg = aggregation_value(cells);
            if (last_agg > peak_agg) peak_agg = last_agg;
            if (record_snapshots) {
                std::vector<int> row(n);
                for (int i = 0; i < n; ++i) row[i] = cells[i].value;
                out.value_snapshots.push_back(row);
            }
        }
    }

    out.swaps = swaps;
    out.comparisons = comparisons;
    out.final_error = monotonicity_error(cells);
    out.dg = delayed_gratification(mono_trace, true);
    out.peak_agg = peak_agg;
    out.final_agg = last_agg;
    out.final_sortedness = sortedness(cells, 1);

    std::vector<int> free_vals;
    out.final_values.resize(n);
    for (int i = 0; i < n; ++i) {
        out.final_values[i] = cells[i].value;
        if (cells[i].frozen == FrozenKind::None) free_vals.push_back(cells[i].value);
    }
    int inv = 0;
    for (size_t i = 0; i < free_vals.size(); ++i) {
        for (size_t j = i + 1; j < free_vals.size(); ++j) {
            if (free_vals[i] > free_vals[j]) ++inv;
        }
    }
    out.free_inversions = inv;
    return out;
}

std::vector<Cell> make_cells(
    const std::vector<int>& values,
    const std::vector<Algotype>& algos,
    const std::vector<int>& frozen_idx = {},
    FrozenKind fkind = FrozenKind::None,
    const std::vector<int>& dirs = {}
) {
    int n = static_cast<int>(values.size());
    std::vector<Cell> cells(n);
    for (int i = 0; i < n; ++i) {
        int d = dirs.empty() ? 1 : dirs[i];
        cells[i] = Cell{
            values[i],
            algos[i],
            FrozenKind::None,
            (d == 1) ? 0 : (n - 1),
            d
        };
    }
    for (int idx : frozen_idx) {
        if (idx >= 0 && idx < n) cells[idx].frozen = fkind;
    }
    return cells;
}

// Traditional top-down baselines (for efficiency ratios)
std::pair<int, int> traditional_sort_counts(std::vector<int> a, Algotype algo) {
    int n = static_cast<int>(a.size());
    int swaps = 0, comps = 0;
    if (algo == Algotype::Bubble) {
        for (int i = 0; i < n; ++i) {
            bool any = false;
            for (int j = 0; j + 1 < n - i; ++j) {
                ++comps;
                if (a[j] > a[j + 1]) {
                    std::swap(a[j], a[j + 1]);
                    ++swaps;
                    ++comps;
                    any = true;
                }
            }
            if (!any) break;
        }
    } else if (algo == Algotype::Insertion) {
        for (int i = 1; i < n; ++i) {
            int j = i;
            while (j > 0) {
                ++comps;
                if (a[j] < a[j - 1]) {
                    std::swap(a[j], a[j - 1]);
                    ++swaps;
                    ++comps;
                    --j;
                } else {
                    break;
                }
            }
        }
    } else {
        for (int i = 0; i < n; ++i) {
            int min_idx = i;
            for (int j = i + 1; j < n; ++j) {
                ++comps;
                if (a[j] < a[min_idx]) min_idx = j;
            }
            if (min_idx != i) {
                std::swap(a[i], a[min_idx]);
                ++swaps;
                ++comps;
            }
        }
    }
    return {swaps, comps};
}

int main(int argc, char** argv) {
    int n = 100;
    int reps = 15;
    uint64_t seed = 42;
    std::string out_json = "sorting-morphogenesis/cpp/cpp_validation_benchmarks.json";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc) n = std::stoi(argv[++i]);
        else if (arg == "--reps" && i + 1 < argc) reps = std::stoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
        else if (arg == "--out" && i + 1 < argc) out_json = argv[++i];
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    // 1. Deterministic Kymograph Equivalence Test (n_kymo = 42, f = 2 at [14, 28])
    const int n_kymo = 42;
    SplitMix64 rng_k(seed);
    std::vector<int> init_k(n_kymo);
    std::iota(init_k.begin(), init_k.end(), 1);
    rng_k.shuffle(init_k);
    std::vector<int> f_pos = {n_kymo / 3, (2 * n_kymo) / 3};
    auto it_hi = std::find(init_k.begin(), init_k.end(), n_kymo - 2);
    std::swap(init_k[f_pos[0]], *it_hi);
    auto it_lo = std::find(init_k.begin(), init_k.end(), 3);
    std::swap(init_k[f_pos[1]], *it_lo);

    auto res_k_b = simulate_cell_view(make_cells(init_k, std::vector<Algotype>(n_kymo, Algotype::Bubble), f_pos, FrozenKind::Immovable), seed + 66, true, true);
    auto res_k_i = simulate_cell_view(make_cells(init_k, std::vector<Algotype>(n_kymo, Algotype::Insertion), f_pos, FrozenKind::Immovable), seed + 73, true, true);
    auto res_k_s = simulate_cell_view(make_cells(init_k, std::vector<Algotype>(n_kymo, Algotype::Selection), f_pos, FrozenKind::Immovable), seed + 83, true, true);

    // 2. Quantitative Paper Replication Suite (n = 100, reps = 15)
    SplitMix64 rng(seed);
    double sum_trad_sw[3] = {0, 0, 0}, sum_cell_sw[3] = {0, 0, 0};
    double sum_trad_tot[3] = {0, 0, 0}, sum_cell_tot[3] = {0, 0, 0};
    Algotype all_algos[3] = {Algotype::Bubble, Algotype::Insertion, Algotype::Selection};

    for (int r = 0; r < reps; ++r) {
        std::vector<int> vals(n);
        std::iota(vals.begin(), vals.end(), 1);
        rng.shuffle(vals);
        for (int a_idx = 0; a_idx < 3; ++a_idx) {
            Algotype a = all_algos[a_idx];
            auto [ts, tc] = traditional_sort_counts(vals, a);
            sum_trad_sw[a_idx] += ts;
            sum_trad_tot[a_idx] += tc;

            uint64_t r_seed = seed * 1000 + r * 7 + static_cast<uint64_t>(a_idx);
            auto rs = simulate_cell_view(make_cells(vals, std::vector<Algotype>(n, a)), r_seed, true);
            sum_cell_sw[a_idx] += rs.swaps;
            if (a == Algotype::Selection) {
                auto rc = simulate_cell_view(make_cells(vals, std::vector<Algotype>(n, a)), r_seed, false);
                sum_cell_tot[a_idx] += rc.comparisons;
            } else {
                sum_cell_tot[a_idx] += rs.comparisons;
            }
        }
    }

    // Immovable Frozen Error Tolerance (f = 3) & Delayed Gratification
    double sum_immov_f3[3] = {0, 0, 0};
    double sum_dg_f3_b = 0.0, sum_dg_i = 0.0;
    for (int r = 0; r < reps; ++r) {
        std::vector<int> vals(n);
        std::iota(vals.begin(), vals.end(), 1);
        rng.shuffle(vals);
        std::vector<int> idx_pool(n);
        std::iota(idx_pool.begin(), idx_pool.end(), 0);
        rng.shuffle(idx_pool);
        std::vector<int> f3 = {idx_pool[0], idx_pool[1], idx_pool[2]};

        for (int a_idx = 0; a_idx < 3; ++a_idx) {
            Algotype a = all_algos[a_idx];
            auto rf = simulate_cell_view(make_cells(vals, std::vector<Algotype>(n, a), f3, FrozenKind::Immovable), seed * 3000 + r * 13 + a_idx);
            sum_immov_f3[a_idx] += rf.final_error;
            if (a == Algotype::Bubble) sum_dg_f3_b += rf.dg;
            if (a == Algotype::Insertion) sum_dg_i += rf.dg;
        }
    }

    // Chimeric Aggregation: Unique vs Duplicate Values (B+S)
    double sum_uniq_peak = 0.0, sum_dup_final = 0.0;
    for (int r = 0; r < reps; ++r) {
        std::vector<int> vu(n), vd(n);
        std::iota(vu.begin(), vu.end(), 1);
        for (int i = 0; i < n; ++i) vd[i] = (i % 10) + 1;
        rng.shuffle(vu);
        rng.shuffle(vd);
        std::vector<Algotype> at(n);
        for (int i = 0; i < n; ++i) at[i] = (i < n / 2) ? Algotype::Bubble : Algotype::Selection;
        rng.shuffle(at);

        auto ru = simulate_cell_view(make_cells(vu, at), seed * 5000 + r * 17);
        auto rd = simulate_cell_view(make_cells(vd, at), seed * 6000 + r * 19);
        sum_uniq_peak += ru.peak_agg;
        sum_dup_final += rd.final_agg;
    }

    // Cross-purpose opposite-direction B(inc)+I(dec) and B(dec)+S(inc)
    int cross_reps = std::max(6, reps / 2);
    double sum_cross_bi = 0.0, sum_cross_bs = 0.0;
    for (int r = 0; r < cross_reps; ++r) {
        std::vector<int> v(n);
        std::iota(v.begin(), v.end(), 1);
        rng.shuffle(v);
        std::vector<Algotype> at_bi(n), at_bs(n);
        std::vector<int> dir_bi(n), dir_bs(n);
        for (int i = 0; i < n; ++i) {
            at_bi[i] = (i < n / 2) ? Algotype::Bubble : Algotype::Insertion;
            dir_bi[i] = (i < n / 2) ? 1 : -1;
            at_bs[i] = (i < n / 2) ? Algotype::Bubble : Algotype::Selection;
            dir_bs[i] = (i < n / 2) ? -1 : 1;
        }
        auto r_bi = simulate_cell_view(make_cells(v, at_bi, {}, FrozenKind::None, dir_bi), seed * 8000 + r * 23, true, false, 120 * n, 120 * n * 20, false);
        auto r_bs = simulate_cell_view(make_cells(v, at_bs, {}, FrozenKind::None, dir_bs), seed * 9000 + r * 29, true, false, 120 * n, 120 * n * 20, false);
        sum_cross_bi += r_bi.final_sortedness * 100.0;
        sum_cross_bs += r_bs.final_sortedness * 100.0;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::ofstream ofs(out_json);
    ofs << std::fixed << std::setprecision(4);
    ofs << "{\n";
    ofs << "  \"language\": \"C++20 (clang++ -std=c++20 -O3)\",\n";
    ofs << "  \"prng\": \"SplitMix64\",\n";
    ofs << "  \"elapsed_ms\": " << elapsed_ms << ",\n";
    ofs << "  \"n\": " << n << ",\n";
    ofs << "  \"reps\": " << reps << ",\n";
    ofs << "  \"kymograph_n42\": {\n";
    ofs << "    \"bubble\": {\"swaps\": " << res_k_b.swaps << ", \"final_error\": " << res_k_b.final_error << ", \"free_inversions\": " << res_k_b.free_inversions << ", \"dg\": " << res_k_b.dg << "},\n";
    ofs << "    \"insertion\": {\"swaps\": " << res_k_i.swaps << ", \"final_error\": " << res_k_i.final_error << ", \"free_inversions\": " << res_k_i.free_inversions << ", \"dg\": " << res_k_i.dg << "},\n";
    ofs << "    \"selection\": {\"swaps\": " << res_k_s.swaps << ", \"final_error\": " << res_k_s.final_error << ", \"free_inversions\": " << res_k_s.free_inversions << ", \"dg\": " << res_k_s.dg << "},\n";
    ofs << "    \"selection_final_values\": [";
    for (int i = 0; i < n_kymo; ++i) ofs << res_k_s.final_values[i] << (i + 1 < n_kymo ? ", " : "");
    ofs << "],\n";
    ofs << "    \"bubble_final_values\": [";
    for (int i = 0; i < n_kymo; ++i) ofs << res_k_b.final_values[i] << (i + 1 < n_kymo ? ", " : "");
    ofs << "]\n";
    ofs << "  },\n";
    ofs << "  \"benchmarks\": {\n";
    ofs << "    \"fig4a_bubble_swaps\": " << (sum_cell_sw[0] / reps) << ",\n";
    ofs << "    \"fig4a_insertion_swaps\": " << (sum_cell_sw[1] / reps) << ",\n";
    ofs << "    \"fig4a_selection_swap_ratio\": " << (sum_cell_sw[2] / std::max(1.0, sum_trad_sw[2])) << ",\n";
    ofs << "    \"fig4b_bubble_step_speedup\": " << (sum_trad_tot[0] / sum_cell_tot[0]) << ",\n";
    ofs << "    \"fig4b_insertion_step_speedup\": " << (sum_trad_tot[1] / sum_cell_tot[1]) << ",\n";
    ofs << "    \"fig4b_selection_step_overhead\": " << (sum_cell_tot[2] / sum_trad_tot[2]) << ",\n";
    ofs << "    \"fig5c_selection_immovable_f3\": " << (sum_immov_f3[2] / reps) << ",\n";
    ofs << "    \"fig5c_bubble_immovable_f3\": " << (sum_immov_f3[0] / reps) << ",\n";
    ofs << "    \"fig7a_bubble_dg_f3\": " << (sum_dg_f3_b / reps) << ",\n";
    ofs << "    \"fig7b_insertion_dg_f3\": " << (sum_dg_i / reps) << ",\n";
    ofs << "    \"fig8a_unique_chimera_peak_agg\": " << (sum_uniq_peak / reps) << ",\n";
    ofs << "    \"fig8d_duplicate_chimera_final_agg\": " << (sum_dup_final / reps) << ",\n";
    ofs << "    \"fig9_cross_purpose_bi_sortedness\": " << (sum_cross_bi / cross_reps) << ",\n";
    ofs << "    \"fig9_cross_purpose_bs_sortedness\": " << (sum_cross_bs / cross_reps) << "\n";
    ofs << "  }\n";
    ofs << "}\n";
    ofs.close();

    std::cout << "[C++20 Port] Completed n=" << n << ", reps=" << reps
              << " in " << std::fixed << std::setprecision(1) << elapsed_ms << " ms -> " << out_json << "\n";
    std::cout << "  Kymograph (n=42, f=2): Bubble(swaps=" << res_k_b.swaps << ", err=" << res_k_b.final_error
              << ", inv=" << res_k_b.free_inversions << ") | Selection(swaps=" << res_k_s.swaps
              << ", err=" << res_k_s.final_error << ", inv=" << res_k_s.free_inversions << ")\n";
    return 0;
}
