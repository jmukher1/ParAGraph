#!/usr/bin/env python3
"""
amazon_to_citation_seed.py

Converts the SNAP "com-amazon.ungraph.txt" co-purchase network into the
seed-graph input format used by citation_abm (sj_nodelist / sj_edgelist /
sj_recprob), so it can be used as a seed graph for the PA/ER growth
simulation instead of the original citation dataset.

WHY THIS ISN'T A DIRECT FORMAT CONVERSION
------------------------------------------
com-amazon.ungraph.txt is an UNDIRECTED co-purchase graph with no
timestamps. The citation ABM needs:
  1. sj_nodelist: node_id,pub_year   -- a publication year per node
  2. sj_edgelist: citing,cited       -- DIRECTED edges (newer cites older)
  3. sj_recprob:  index,prob         -- P(cite a paper with this year_diff)

None of that exists in the raw Amazon data, so this script SYNTHESIZES it:
  - Publication years are sampled per node from a configurable distribution
    over [--start-year, --end-year], skewed so later years get more nodes
    (mimicking the real growth-over-time shape seen in citation datasets
    like sj_nodelist, where counts-per-year increase toward the present).
  - Each undirected edge (u, v) is ORIENTED using the assigned years: the
    node with the LATER year "cites" the node with the EARLIER year, which
    is the standard citation-DAG convention (you can't cite something
    published after you). Ties (same year) are broken deterministically by
    node id so the output is reproducible for a fixed --seed.
  - The recency probability curve is a synthetic log-normal-shaped decay
    over year_diff bins, normalized to sum to 1 -- matching the general
    shape of real citation-aging curves (a short rise for very recent
    work, then decay for older work), NOT a fit to any specific real
    dataset. Tune --recency-peak / --recency-sigma if you want a sharper
    or flatter aging profile.

This is a clearly-labeled synthetic construction, not a recovery of any
real ground-truth structure in the Amazon graph -- treat outputs as a
stress-test/benchmark seed graph (useful for scale and structural
diversity), not as a dataset with real bibliometric meaning.

USAGE
-----
    python3 amazon_to_citation_seed.py \\
        --input com-amazon_ungraph.txt \\
        --out-prefix amz \\
        --start-year 1950 --end-year 2020 \\
        --recency-bins 356 \\
        --min-degree 0 \\
        --seed 42

Produces: amz_nodelist, amz_edgelist, amz_recprob
"""

import argparse
import math
import random
import sys
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
                   help="Path to com-amazon.ungraph.txt (or any SNAP-format "
                        "'FromNodeId TAB ToNodeId' undirected edge list, "
                        "'#'-prefixed comment lines allowed)")
    p.add_argument("--out-prefix", required=True,
                   help="Output file prefix; writes <prefix>_nodelist, "
                        "<prefix>_edgelist, <prefix>_recprob")
    p.add_argument("--start-year", type=int, default=1950,
                   help="Earliest synthetic publication year (default: 1950)")
    p.add_argument("--end-year", type=int, default=2020,
                   help="Latest synthetic publication year (default: 2020)")
    p.add_argument("--growth-shape", type=float, default=3.0,
                   help="Skew toward later years: higher = more nodes in "
                        "later years, matching real citation-dataset growth "
                        "curves. 1.0 = uniform over the year range. "
                        "(default: 3.0)")
    p.add_argument("--recency-bins", type=int, default=356,
                   help="Number of year_diff bins in the recency probability "
                        "table (default: 356, matching the reference "
                        "sj_recprob file's size)")
    p.add_argument("--recency-peak", type=float, default=2.0,
                   help="year_diff at which citation probability peaks "
                        "(default: 2.0, i.e. papers are most often cited "
                        "~2 years after publication)")
    p.add_argument("--recency-sigma", type=float, default=1.2,
                   help="Log-normal spread of the recency curve; higher = "
                        "flatter/broader aging profile (default: 1.2)")
    p.add_argument("--min-degree", type=int, default=0,
                   help="Drop nodes with total (undirected) degree below "
                        "this threshold before conversion (default: 0, "
                        "i.e. keep all nodes)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible year assignment and "
                        "tie-breaking (default: 42)")
    return p.parse_args()


def read_edges(path):
    """Read SNAP-format undirected edge list, skipping '#' comment lines."""
    edges = []
    nodes = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u == v:
                continue  # drop self-loops defensively, even though none were found in com-amazon
            edges.append((u, v))
            nodes.add(u)
            nodes.add(v)
    return edges, nodes


def filter_by_degree(edges, nodes, min_degree):
    """Drop nodes below min_degree, and any edge touching a dropped node."""
    if min_degree <= 0:
        return edges, nodes
    degree = defaultdict(int)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    keep = {n for n in nodes if degree[n] >= min_degree}
    filtered_edges = [(u, v) for (u, v) in edges if u in keep and v in keep]
    return filtered_edges, keep


def assign_years(nodes, start_year, end_year, growth_shape, rng):
    """
    Sample a publication year per node from a growth-skewed distribution
    over [start_year, end_year]. growth_shape > 1 concentrates more nodes
    in later years (x^growth_shape inverse-CDF sampling), matching the
    real-world pattern of citation datasets having far more recent papers
    than old ones.
    """
    span = end_year - start_year
    if span <= 0:
        raise ValueError("--end-year must be greater than --start-year")
    years = {}
    for n in nodes:
        u = rng.random()
        # inverse-CDF of a Beta-like skew: pushes mass toward u=1 (later years)
        # for growth_shape > 1, uniform for growth_shape == 1
        frac = u ** (1.0 / growth_shape)
        years[n] = start_year + int(round(frac * span))
    return years


def orient_edges(edges, years):
    """
    Orient each undirected edge from the later-year node (citing) to the
    earlier-year node (cited). Ties broken by node id (larger id "cites"
    smaller id) so output is deterministic given a fixed year assignment.
    """
    directed = []
    for u, v in edges:
        yu, yv = years[u], years[v]
        if yu > yv or (yu == yv and u > v):
            citing, cited = u, v
        else:
            citing, cited = v, u
        directed.append((citing, cited))
    return directed


def build_recency_table(num_bins, peak, sigma):
    """
    Synthetic log-normal-shaped recency curve over year_diff in
    [0, num_bins-1], normalized to sum to 1. Uses (year_diff + 1) as the
    log-normal input to avoid log(0) at year_diff == 0.

    The log-normal distribution's MODE (peak location) is exp(mu - sigma^2),
    NOT exp(mu) -- so mu is solved backwards from the desired peak location
    (mode = peak + 1 in x-space) rather than set directly to log(peak + 1).
    Getting this wrong silently shifts the peak toward year_diff=0 (or even
    negative, effectively monotonic decay) instead of the intended rise-
    then-decay shape.
    """
    mode_x = peak + 1.0  # desired mode in x = (year_diff + 1) space
    mu = math.log(mode_x) + sigma ** 2
    probs = []
    for d in range(num_bins):
        x = d + 1.0
        val = (1.0 / (x * sigma * math.sqrt(2 * math.pi))) * \
              math.exp(-((math.log(x) - mu) ** 2) / (2 * sigma ** 2))
        probs.append(val)
    total = sum(probs)
    return [p / total for p in probs]


def write_nodelist(path, years):
    with open(path, "w") as f:
        f.write("# node_id,pub_year\n")
        for node_id in sorted(years.keys()):
            f.write(f"{node_id},{years[node_id]}\n")


def write_edgelist(path, directed_edges):
    with open(path, "w") as f:
        f.write("# citing,cited\n")
        for citing, cited in directed_edges:
            f.write(f"{citing},{cited}\n")


def write_recprob(path, probs):
    with open(path, "w") as f:
        f.write("#index,prob\n")
        for i, p in enumerate(probs):
            f.write(f"{i},{p}\n")


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"Reading edges from {args.input} ...", file=sys.stderr)
    edges, nodes = read_edges(args.input)
    print(f"  {len(nodes):,} nodes, {len(edges):,} edges (undirected)", file=sys.stderr)

    if args.min_degree > 0:
        edges, nodes = filter_by_degree(edges, nodes, args.min_degree)
        print(f"  after --min-degree {args.min_degree}: "
              f"{len(nodes):,} nodes, {len(edges):,} edges", file=sys.stderr)

    print(f"Assigning synthetic publication years "
          f"[{args.start_year}, {args.end_year}], growth_shape={args.growth_shape} ...",
          file=sys.stderr)
    years = assign_years(nodes, args.start_year, args.end_year, args.growth_shape, rng)

    print("Orienting edges by year (citing = later year, cited = earlier year) ...",
          file=sys.stderr)
    directed_edges = orient_edges(edges, years)

    print(f"Building recency probability table "
          f"({args.recency_bins} bins, peak={args.recency_peak}, sigma={args.recency_sigma}) ...",
          file=sys.stderr)
    probs = build_recency_table(args.recency_bins, args.recency_peak, args.recency_sigma)

    nodelist_path = f"{args.out_prefix}_nodelist"
    edgelist_path = f"{args.out_prefix}_edgelist"
    recprob_path = f"{args.out_prefix}_recprob"

    write_nodelist(nodelist_path, years)
    write_edgelist(edgelist_path, directed_edges)
    write_recprob(recprob_path, probs)

    print(f"\nWrote:\n  {nodelist_path}  ({len(years):,} nodes)"
          f"\n  {edgelist_path}  ({len(directed_edges):,} directed edges)"
          f"\n  {recprob_path}  ({len(probs)} bins, sums to "
          f"{sum(probs):.6f})", file=sys.stderr)


if __name__ == "__main__":
    main()
