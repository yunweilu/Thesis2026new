"""
DRAG (Derivative Removal by Adiabatic Gate) pulse simulation.

System Hamiltonian (rotating frame, Kerr nonlinearity):
    H₀ = (K/2) b†² b²

Two-quadrature drive:
    H_c(t) = Ωx(t) · (b + b†)/2  +  Ωy(t) · i(b† − b)/2

DRAG pulse scheme:
    Ωx(t) = A · exp(−(t − t₀)² / (2σ²))       [Gaussian envelope]
    Ωy(t) = −λ · (dΩx/dt) / K                   [DRAG Y-quadrature correction]

Reference: Motzoi et al., PRL 103, 110501 (2009)
"""

import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════
# Plot style  (from polish_plot)
# ══════════════════════════════════════════════════════════════
fig_width_pt  = 246.0
inches_per_pt = 1.0 / 72.27
fig_width     = fig_width_pt * inches_per_pt
fig_height    = fig_width / 1.45

normal_plot = {
    "figure.figsize":      (fig_width * 2, fig_height * 3),
    "figure.dpi":          150,
    "savefig.dpi":         600,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.02,
    "font.family":         "STIXGeneral",
    "font.size":           8.5,
    "axes.labelsize":      9,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "legend.fontsize":     8,
    "mathtext.fontset":    "stix",
    "lines.linewidth":     1.6,
    "lines.markersize":    4.0,
    "axes.linewidth":      0.8,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.major.size":    3.5,
    "ytick.major.size":    3.5,
    "xtick.major.width":   0.8,
    "ytick.major.width":   0.8,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.size":    2.0,
    "ytick.minor.size":    2.0,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "legend.frameon":      False,
    "axes.grid":           False,
}
mpl.rcParams.update(normal_plot)
mpl.rcParams['svg.fonttype'] = 'path'
mpl.rcParams['pdf.fonttype'] = 42


# ══════════════════════════════════════════════════════════════
# System parameters
# ══════════════════════════════════════════════════════════════
N_dim  = 4                          # Hilbert-space truncation
K      = -0.20 * 2 * np.pi         # Kerr coefficient  (rad/ns, i.e. −200 MHz)
t_gate = 15.0                     # total gate duration  (ns)
sigma  = t_gate / 6                 # Gaussian standard deviation
t0     = t_gate / 2                  # pulse centre
lam    = 1.0                         # DRAG scaling (1 = first-order DRAG)

# π-pulse amplitude:  ∫ Ωx(t) dt = π  →  A = π / (σ √(2π))
A = np.pi / (sigma * np.sqrt(2 * np.pi))

# time grid
N_t   = 1000
tlist = np.linspace(0, t_gate, N_t)


# ══════════════════════════════════════════════════════════════
# Operators
# ══════════════════════════════════════════════════════════════
b    = qt.destroy(N_dim)
bdag = b.dag()
n_op = bdag * b

# Static Hamiltonian  (Kerr)
H_kerr = (K / 2) * bdag * bdag * b * b           # (K/2) b†² b²

# Drive operators
H_x = (b + bdag) / 2                              # in-phase  (X quadrature)
H_y = 1j * (bdag - b) / 2                         # out-of-phase (Y quadrature)


# ══════════════════════════════════════════════════════════════
# Pulse envelopes
# ══════════════════════════════════════════════════════════════
def gaussian_envelope(t, A, t0, sigma):
    """Gaussian pulse on the X quadrature."""
    return A * np.exp(-(t - t0)**2 / (2 * sigma**2))


def drag_y_envelope(t, A, t0, sigma, K, lam=0.5):
    """
    DRAG correction on the Y quadrature:
        Ωy(t) = −λ · dΩx/dt / K
    """
    gauss   = gaussian_envelope(t, A, t0, sigma)
    d_gauss = -(t - t0) / sigma**2 * gauss        # dΩx/dt
    return -lam * d_gauss / K


# QuTiP coefficient callbacks
def _coeff_Omega_x(t, args):
    return gaussian_envelope(t, args['A'], args['t0'], args['sigma'])


def _coeff_Omega_y(t, args):
    return drag_y_envelope(
        t, args['A'], args['t0'], args['sigma'], args['K'], args['lam']
    )


# ══════════════════════════════════════════════════════════════
# Helper: simulate with a given detuning and DRAG coefficient
# ══════════════════════════════════════════════════════════════
H_detune = n_op                                    # δ · b†b

def run_sim(delta, use_drag, lam_val=lam, return_states=False):
    """Simulate with detuning δ and DRAG coefficient λ."""
    pulse_args = {'A': A, 't0': t0, 'sigma': sigma, 'K': K, 'lam': lam_val}
    H_static = H_kerr + delta * H_detune

    if use_drag:
        H = [H_static, [H_x, _coeff_Omega_x], [H_y, _coeff_Omega_y]]
    else:
        H = [H_static, [H_x, _coeff_Omega_x]]

    psi0   = qt.basis(N_dim, 0)
    result = qt.sesolve(H, psi0, tlist, args=pulse_args)

    if return_states:
        return result

    psi_f = result.states[-1]
    return abs(psi_f.overlap(qt.basis(N_dim, 1)))**2


# ══════════════════════════════════════════════════════════════
# Optimize detuning (no-DRAG) and (δ, λ) jointly (DRAG)
# ══════════════════════════════════════════════════════════════
from scipy.optimize import minimize_scalar, minimize

# no-DRAG: optimize δ only
print("Optimizing δ for no-DRAG case …")
res_no = minimize_scalar(lambda d: -run_sim(d, use_drag=False),
                         bounds=(-0.5 * 2*np.pi, 0.5 * 2*np.pi),
                         method='bounded')
delta_opt_no = res_no.x
print(f"  δ_opt/(2π) = {delta_opt_no/(2*np.pi):.4f} GHz,  P(|1⟩) = {-res_no.fun:.4f}")

# DRAG: jointly optimize δ and λ
print("Optimizing (δ, λ) for DRAG case …")
def drag_cost(x):
    d, l = x
    return -run_sim(d, use_drag=True, lam_val=l)

res_dr = minimize(drag_cost, x0=[0.0, 1.0], method='Nelder-Mead',
                  options={'xatol': 1e-4, 'fatol': 1e-6})
delta_opt_drag, lam_opt = res_dr.x
print(f"  δ_opt/(2π) = {delta_opt_drag/(2*np.pi):.4f} GHz,  λ_opt = {lam_opt:.4f},  P(|1⟩) = {-res_dr.fun:.4f}")

# Final simulations with optimal parameters
print("Running final simulations …")
result_no_drag = run_sim(delta_opt_no,   use_drag=False, return_states=True)
result_drag    = run_sim(delta_opt_drag, use_drag=True, lam_val=lam_opt, return_states=True)


# ══════════════════════════════════════════════════════════════
# Extract populations  (|0⟩, |1⟩, |2⟩ only)
# ══════════════════════════════════════════════════════════════
def extract_populations(result, levels=(0, 1, 2)):
    """Return dict  {n: array} with |⟨n|ψ(t)⟩|²."""
    pops = {}
    for n in levels:
        pops[n] = np.array([abs(psi.overlap(qt.basis(N_dim, n)))**2
                            for psi in result.states])
    return pops


pops_no_drag = extract_populations(result_no_drag)
pops_drag    = extract_populations(result_drag)


# ══════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════
colors = {
    'x_pulse':  '#2563EB',   # blue
    'y_pulse':  '#DC2626',   # red
    0:          '#2563EB',   # blue   — |0⟩
    1:          '#DC2626',   # red    — |1⟩
    2:          '#16A34A',   # green  — |2⟩ (leakage)
}
labels = {0: r'$|0\rangle$', 1: r'$|1\rangle$', 2: r'$|2\rangle$'}

fig, axes = plt.subplots(2, 2, figsize=(fig_width * 2.4, fig_height * 2.4),
                         gridspec_kw={'hspace': 0.30, 'wspace': 0.30})

# ── (a) Gaussian-only control ─────────────────────────────────
ax = axes[0, 0]
Ox = np.array([gaussian_envelope(t, A, t0, sigma) for t in tlist])
ax.plot(tlist, Ox / (2 * np.pi), color=colors['x_pulse'],
        label=r'$\Omega_x(t)$')
ax.axhline(0, lw=0.5, color='grey', zorder=0)
ax.set_ylabel(r'Amplitude (GHz)')
ax.set_xlabel('Time (ns)')
ax.legend(loc='upper right')
ax.set_title('Control — no DRAG', fontsize=9)
ax.text(0.02, 0.92, '(a)', transform=ax.transAxes, fontweight='bold', va='top')

# ── (b) Population — no DRAG ──────────────────────────────────
ax = axes[0, 1]
for n in (0, 1, 2):
    ax.plot(tlist, pops_no_drag[n], color=colors[n], label=labels[n])
ax.set_ylabel('Population')
ax.set_xlabel('Time (ns)')
ax.legend(loc='center right')
ax.set_ylim(-0.05, 1.05)
ax.set_title(
    rf'no DRAG ($\delta/2\pi = {delta_opt_no/(2*np.pi):.3f}$ GHz)',
    fontsize=9,
)
ax.text(0.02, 0.92, '(b)', transform=ax.transAxes, fontweight='bold', va='top')

# ── (c) DRAG control ──────────────────────────────────────────
ax = axes[1, 0]
Oy = np.array([drag_y_envelope(t, A, t0, sigma, K, lam_opt) for t in tlist])
ax.plot(tlist, Ox / (2 * np.pi), color=colors['x_pulse'],
        label=r'$\Omega_x(t)$')
ax.plot(tlist, Oy / (2 * np.pi), color=colors['y_pulse'],
        label=r'$\Omega_y(t)$', ls='--')
ax.axhline(0, lw=0.5, color='grey', zorder=0)
ax.set_ylabel(r'Amplitude (GHz)')
ax.set_xlabel('Time (ns)')
ax.legend(loc='upper right')
ax.set_title('Control — with DRAG', fontsize=9)
ax.text(0.02, 0.92, '(c)', transform=ax.transAxes, fontweight='bold', va='top')

# ── (d) Population — with DRAG ────────────────────────────────
ax = axes[1, 1]
for n in (0, 1, 2):
    ax.plot(tlist, pops_drag[n], color=colors[n], label=labels[n])
ax.set_ylabel('Population')
ax.set_xlabel('Time (ns)')
ax.legend(loc='center right')
ax.set_ylim(-0.05, 1.05)
ax.set_title(
    rf'with DRAG ($\delta/2\pi = {delta_opt_drag/(2*np.pi):.3f}$ GHz, $\lambda = {lam_opt:.2f}$)',
    fontsize=9,
)
ax.text(0.02, 0.92, '(d)', transform=ax.transAxes, fontweight='bold', va='top')

# ── Summary ───────────────────────────────────────────────────
leak_no = pops_no_drag[2][-1]
leak_dr = pops_drag[2][-1]
p1_no   = pops_no_drag[1][-1]
p1_dr   = pops_drag[1][-1]
print(f"\n=== Final-state comparison ===")
print(f"  no DRAG  (δ/(2π) = {delta_opt_no/(2*np.pi):.4f} GHz):  P(|1⟩) = {p1_no:.4f},  P(|2⟩) = {leak_no:.4e}")
print(f"  DRAG     (δ/(2π) = {delta_opt_drag/(2*np.pi):.4f} GHz, λ = {lam_opt:.4f}):  P(|1⟩) = {p1_dr:.4f},  P(|2⟩) = {leak_dr:.4e}")

plt.savefig('drag_pulse.pdf')
print("\nSaved drag_pulse.pdf")
