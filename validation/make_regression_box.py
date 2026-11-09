"""Deterministic synthetic periodic box for OASIS regression testing.

Usage:
    python make_regression_box.py small out.npz    # ~4.1M particles, 1,375 seeds
    python make_regression_box.py dense out.npz    # ~59M particles, 21,246 seeds

The box (100 Mpc/h, particle mass 1e10) contains NFW-like haloes with infall
shells, subhaloes that should and should not be absorbed, close pairs that
compete in percolation, fake seeds with almost no particles, quantized masses
(so M200b ties occur), haloes on mini-box and periodic edges, and a uniform
background. Random seeds are fixed, so the same command always produces the
same box, byte for byte.
"""
import sys

import numpy as np

G = 4.3e-09
BOX = 100.0
MP = 1e10
RHOM = 0.3 * 2.775e11

# name -> (random seed, n_hosts, log10 max host mass, subhalo probability, max subhaloes + 1)
PRESETS = {
    'small': (1, 600, 14.3, 0.5, 4),
    'dense': (2, 2500, 14.8, 0.9, 25),
}


def nfw_radii(rng, n, rs, rmax):
    # inverse CDF of NFW enclosed mass, numerically
    x = np.linspace(1e-4, rmax / rs, 4000)
    m = np.log(1 + x) - x / (1 + x)
    u = rng.uniform(0, m[-1], n)
    return np.interp(u, m, x) * rs


def iso(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


def make_box(out, seed=1, n_hosts=600, logm_max=14.3, sub_prob=0.5, nsub_max=4):
    rng = np.random.default_rng(seed)
    pos, vel = [], []
    s_pos, s_vel, s_m, s_r, s_rs = [], [], [], [], []

    def add_halo(center, bulk, M, sigma_fac=0.6, infall=True):
        r200 = (3 * M / (4 * np.pi * 200 * RHOM)) ** (1 / 3)
        c = rng.uniform(4, 10)
        rs = r200 / c
        v200 = np.sqrt(G * M / r200)
        n = int(M / MP)
        r = nfw_radii(rng, n, rs, 1.2 * r200)
        p = center + iso(rng, n) * r[:, None]
        v = bulk + rng.normal(scale=sigma_fac * v200, size=(n, 3))
        pos.append(p); vel.append(v)
        if infall:
            ni = int(0.3 * n)
            ri = rng.uniform(1.0, 2.5, ni) * r200
            d = iso(rng, ni)
            pos.append(center + d * ri[:, None])
            vel.append(bulk - 1.3 * v200 * d + rng.normal(scale=0.3 * v200, size=(ni, 3)))
        s_pos.append(center); s_vel.append(bulk); s_m.append(M); s_r.append(r200); s_rs.append(rs)
        return r200, v200

    logm = rng.uniform(12, logm_max, n_hosts)
    # Quantize masses to multiples of 10 particles -> many exact M200b ties.
    masses = np.round(10 ** logm / (10 * MP)) * 10 * MP
    centers = rng.uniform(0, BOX, (n_hosts, 3))
    # Push some halos onto mini-box (25 Mpc) and periodic boundaries.
    centers[:60, 0] = rng.choice([0.2, 24.9, 25.1, 50.0, 99.8], 60)
    for k in range(n_hosts):
        bulk = rng.normal(scale=300, size=3)
        r200, v200 = add_halo(centers[k], bulk, masses[k])
        if masses[k] > 1e13 and rng.uniform() < sub_prob:
            for _ in range(rng.integers(1, nsub_max)):
                msub = np.round(masses[k] * rng.uniform(0.02, 0.15) / (10 * MP)) * 10 * MP
                off = iso(rng, 1)[0] * rng.uniform(0.2, 1.2) * r200
                fast = rng.uniform() < 0.3
                vsub = bulk + (iso(rng, 1)[0] * 2.5 * v200 if fast
                               else rng.normal(scale=0.5 * v200, size=3))
                add_halo(centers[k] + off, vsub, msub, infall=False)
        # Close pairs for percolation competition.
        if rng.uniform() < 0.1:
            add_halo(centers[k] + iso(rng, 1)[0] * 1.5 * r200,
                     bulk + rng.normal(scale=200, size=3), masses[k] * 0.5)

    # Fake seeds with very few particles around them.
    n_fake = 400
    for _ in range(n_fake):
        M = np.round(10 ** rng.uniform(11, 12) / MP) * MP
        c = rng.uniform(0, BOX, 3)
        r200 = (3 * M / (4 * np.pi * 200 * RHOM)) ** (1 / 3)
        s_pos.append(c); s_vel.append(rng.normal(scale=300, size=3))
        s_m.append(M); s_r.append(r200); s_rs.append(r200 / 5)

    nbg = 1_000_000
    pos.append(rng.uniform(0, BOX, (nbg, 3)))
    vel.append(rng.normal(scale=300, size=(nbg, 3)))

    pos = np.mod(np.concatenate(pos), BOX)
    # Keep positions strictly below BOX after the float32 cast done on write,
    # so the reference code (which crashes on positions equal to BOX) can run.
    pos = np.minimum(pos, float(np.nextafter(np.float32(BOX), np.float32(0))))
    vel = np.concatenate(vel)
    pid = rng.permutation(len(pos)).astype(np.int64) + 1
    s_pos = np.mod(np.array(s_pos), BOX)
    s_pos = np.minimum(s_pos, float(np.nextafter(np.float32(BOX), np.float32(0))))
    hid = rng.permutation(len(s_pos)).astype(np.int64)  # includes ID 0
    np.savez(out, pos=pos, vel=vel, pid=pid, s_pos=s_pos, s_vel=np.array(s_vel),
             s_m=np.array(s_m), s_r=np.array(s_r), s_rs=np.array(s_rs), hid=hid)
    print(f"particles={len(pos)} seeds={len(s_pos)}")


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in PRESETS:
        sys.exit(f"usage: python {sys.argv[0]} {{{'|'.join(PRESETS)}}} out.npz")
    seed, n_hosts, logm_max, sub_prob, nsub_max = PRESETS[sys.argv[1]]
    make_box(sys.argv[2], seed=seed, n_hosts=n_hosts, logm_max=logm_max,
             sub_prob=sub_prob, nsub_max=nsub_max)
