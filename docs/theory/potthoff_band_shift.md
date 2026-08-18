# The Potthoff-Wegner-Nolting band-shift correction

This page derives the optional `use_potthoff_band_shift` correction in full,
and connects it back to the code that implements it
({py:meth}`~neq_kk_ipt_solver.Solver._calc_band_shift_weiss`,
{py:meth}`~neq_kk_ipt_solver.Solver._calc_band_shift_interacting`,
{py:meth}`~neq_kk_ipt_solver.Solver._build_state`,
{py:meth}`~neq_kk_ipt_solver.Solver.calc_SE`) and to the sum rule it exists
to satisfy ({py:mod}`neq_kk_ipt_solver.moments`).

## What it is for

The base KK-IPT-n0 Ansatz (see {py:meth}`~neq_kk_ipt_solver.Solver.calc_SE`)
gets the first two exact spectral moments $M_1^\sigma, M_2^\sigma$ right by
construction — they close on the self-consistent occupations alone. The
third moment $M_3^\sigma$ does not: its closed form needs a genuinely
two-particle quantity, the correlator

$$
C_{\bar\sigma} \equiv n_{\bar\sigma}(1-n_{\bar\sigma})\big(\tilde B_{\bar\sigma} - \epsilon_{f,\bar\sigma}\big),
$$

which the plain scheme has no mechanism to enforce, and violates
substantially (order-10% relative deviation already at moderate $U$, growing
with coupling). The band-shift correction restores $M_3^\sigma$ to the
numerical floor by construction: it self-consistently equates
$\tilde B_{\bar\sigma}$ with exactly the functional in
{eq}`btilde-interacting` below, closing the loop between the self-energy and
the correlator it is supposed to reproduce.

It is included as an *available refinement*, not as part of the base Ansatz
benchmarked in the main text — see ["Accuracy and
limitations"](../index.md#accuracy-and-limitations) for what it does and
does not fix (it changes nothing in the paramagnetic case, and does not
resolve the spin-dependent majority-spin crossover at strong coupling).

Reference: M. Potthoff, T. Wegner, and W. Nolting, *Interpolating
self-energy of the infinite-dimensional Hubbard model: Modifying the
iterative perturbation theory*, [Phys. Rev. B 55, 16132
(1997)](https://doi.org/10.1103/PhysRevB.55.16132).

## Two band shifts, not one

The correction is built from two distinct, energy-dimensioned effective
levels of the *opposite* spin: a Weiss-field (Hartree-Fock-level) estimate
$\tilde B_{0,\bar\sigma}$, and its self-consistent, interacting counterpart
$\tilde B_{\bar\sigma}$. Neither should be confused with the coefficient
$\beta_\sigma$ itself.

**Weiss-field estimate.** A functional of $\mathcal{G}_{0,\bar\sigma}$ and
$\Delta_{\bar\sigma}$ alone — no self-energy needed, so it is cheap and is
recomputed once per trial state:

$$
\tilde B_{0,\bar\sigma} = \epsilon_{f,\bar\sigma} + \frac{2n_{0,\sigma}-1}{n_{0,\bar\sigma}(1-n_{0,\bar\sigma})}\,
(-i)\int\frac{d\omega}{2\pi}\big[\mathcal{G}_{0,\bar\sigma}\Delta_{\bar\sigma}\big]^<(\omega)
$$

with the lesser component expanded via the Langreth rules,

$$
\big[\mathcal{G}_{0,\bar\sigma}\Delta_{\bar\sigma}\big]^<(\omega) =
\mathcal{G}_{0,\bar\sigma}^R(\omega)\Delta_{\bar\sigma}^<(\omega) + \mathcal{G}_{0,\bar\sigma}^<(\omega)\Delta_{\bar\sigma}^A(\omega).
$$

Implemented in
{py:meth}`~neq_kk_ipt_solver.Solver._calc_band_shift_weiss`.

**Interacting counterpart.** Additionally requires the self-energy and
interacting Green's function of species $\bar\sigma$ at the *current* trial
state:

```{math}
:label: btilde-interacting

\tilde B_{\bar\sigma} = \epsilon_{f,\bar\sigma} + \frac{1}{n_{\bar\sigma}(1-n_{\bar\sigma})}\,
(-i)\int\frac{d\omega}{2\pi}\big[Q_{\bar\sigma} G_{\bar\sigma}\Delta_{\bar\sigma}\big]^<(\omega),
\qquad Q_{\bar\sigma} \equiv \frac{2\Sigma_{\bar\sigma}}{U} - 1
```

with the lesser component of the *triple* product expanded via the Langreth rules,

$$
\big[Q_{\bar\sigma} G_{\bar\sigma}\Delta_{\bar\sigma}\big]^< =
Q_{\bar\sigma}^R G_{\bar\sigma}^R \Delta_{\bar\sigma}^< + Q_{\bar\sigma}^R G_{\bar\sigma}^< \Delta_{\bar\sigma}^A
+ Q_{\bar\sigma}^< G_{\bar\sigma}^A \Delta_{\bar\sigma}^A,
$$

where $Q_{\bar\sigma}^R = 2\Sigma_{\bar\sigma}^R/U - 1$ and
$Q_{\bar\sigma}^< = 2\Sigma_{\bar\sigma}^</U$. Implemented in
{py:meth}`~neq_kk_ipt_solver.Solver._calc_band_shift_interacting`.

## The corrected $\beta_\sigma$

The correction enters $\beta_\sigma$ (not $\alpha_\sigma$) as an *additive*
term in the numerator, replacing the plain coefficient:

$$
\beta_\sigma = \frac{(1-n_{\bar\sigma})U + \epsilon_{f,\sigma} + \mu_{0,\sigma}
+ \big(\tilde B_{\bar\sigma} - \tilde B_{0,\bar\sigma}\big)}{n_{0,\bar\sigma}(1-n_{0,\bar\sigma})U^2}
$$

Note that $\epsilon_{f,\bar\sigma}$ never itself appears in the plain
$\beta_\sigma$ (only the own-spin level $\epsilon_{f,\sigma}$ does): this is
a genuinely new term, not a substitution of one already present. It
vanishes, and the plain coefficient is recovered exactly, whenever
$\tilde B_{\bar\sigma} = \tilde B_{0,\bar\sigma}$ — in particular at the seed
of the inner iteration below, and identically in the paramagnetic case
($n_{0,\uparrow}=n_{0,\downarrow}$), which is why the correction is
benchmarked there as changing nothing (see the main README).

Implemented in {py:meth}`~neq_kk_ipt_solver.Solver.calc_SE`.

## The inner fixed-point loop

Because {eq}`btilde-interacting` requires $\Sigma_{\bar\sigma}$ and
$G_{\bar\sigma}$, which themselves depend on $\beta_{\bar\sigma}$ through the
corrected formula above, $\tilde B_{\bar\sigma}$ cannot be evaluated in one
shot — it has to be found by fixed-point iteration *inside* every trial
state visited by the outer four-dimensional root solve
({py:meth}`~neq_kk_ipt_solver.Solver.solve`), not once per converged
solution. This is what
{py:meth}`~neq_kk_ipt_solver.Solver._build_state` does when
`use_potthoff_band_shift` is `True`:

1. Build $\mathcal{G}_{0,\sigma}$ for both flavors from the trial
   $(\mu_{0,\sigma}, n_\sigma)$, and the bare bubble diagram
   ({py:meth}`~neq_kk_ipt_solver.Solver._calc_IPT_diagram`) — these depend
   only on $\mathcal{G}_0$, so computed once and reused for every inner
   iteration below.
2. Compute $\tilde B_{0,\sigma}$ for both flavors
   ({py:meth}`~neq_kk_ipt_solver.Solver._calc_band_shift_weiss`), and seed
   $\tilde B_\sigma \leftarrow \tilde B_{0,\sigma}$ — so the *first* inner
   iteration reduces exactly to the plain coefficient.
3. Repeat until $\max_\sigma|\tilde B_\sigma^{\text{new}} - \tilde
   B_\sigma| <$ `band_shift_tol`, or `band_shift_inner_maxiter` is reached:
   - build $\beta_\sigma$ (with the current $\tilde B$'s) and $\Sigma_\sigma$
     for both flavors ({py:meth}`~neq_kk_ipt_solver.Solver.calc_SE`);
   - build $G_\sigma$ for both flavors
     ({py:meth}`~neq_kk_ipt_solver.Solver.calc_GF`);
   - compute $\tilde B_\sigma^{\text{new}}$ from $\Sigma_\sigma, G_\sigma$
     ({py:meth}`~neq_kk_ipt_solver.Solver._calc_band_shift_interacting`);
   - linearly mix: $\tilde B_\sigma \leftarrow (1-\eta)\tilde B_\sigma +
     \eta\,\tilde B_\sigma^{\text{new}}$, with $\eta =$
     `band_shift_mixing`.
4. Rebuild $\Sigma_\sigma, G_\sigma$ one final time with the converged
   $\tilde B_\sigma$.

Trial states probed by the outer root-finder can carry an occupation
pathologically close to $0$ or $1$, which blows up the $n(1-n)$ denominators
above; if any $\tilde B_\sigma^{\text{new}}$ becomes non-finite or exceeds a
physically motivated bound, the inner loop falls back to the Weiss-field
estimate $\tilde B_{0,\sigma}$ for that trial state rather than propagating a
runaway value (`band_shift_diverged_count` tracks how often this happens —
worth checking after a solve that used this correction).

## Verifying it

{py:func}`neq_kk_ipt_solver.moments.check_sum_rules` compares the closed-form
$M_3^\sigma$ (built from $C_{\bar\sigma}$, i.e. from $\tilde B_{\bar\sigma}$
via {py:func}`~neq_kk_ipt_solver.moments.band_shift_correlator`) against the
numerically integrated $\int d\omega\,\omega^3 A_\sigma(\omega)$. With the
correction on, the two agree to the numerical floor; with it off, plain
KK-IPT-n0 violates $M_3$ by a few percent to tens of percent depending on the
setup. See `scripts/check_spectral_moments.py` for the full comparison this
claim is based on.
