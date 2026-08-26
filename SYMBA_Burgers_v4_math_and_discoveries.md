# SYMBA Burgers Benchmark v4 — Full Mathematics, Bug Post-Mortem, and Discoveries

Final result (full budget, N=1000, S=512, T=20, 100 epochs, identical architecture/training for
every arm): **shared baseline 0.0099 vs Galilean closed-form 0.0009 — 11x better (+90.7%)**,
with TOP-2 merged (Galilean-cf + Reflection) at 0.0009 (+90.8%).

---

## Part 1 — What was wrong in the old math (post-mortem of 3 bugs + 1 design flaw)

### Bug 1 (critical): Galilean time-unit error
The co-moving frame converts the drift to pixels as

    c_px(t) = U * t * (S / L)            # WRONG in v1/v2

treating `t` — the **output-step index** (1..20) — as physical time. One output step is
`dt_out = dt * substeps = 5e-4 * 300 = 0.15` time units, so every frame was over-shifted by
`1/dt_out = 6.67x` the true drift.

**Consequences:** canonical frames still contained violent residual translation (temporal
variance across frames *increased* ~250% instead of dropping), the shape-FNO was asked to fit
scrambled targets (oracle stuck at 0.133 regardless of projector exactness), and every
Galilean-composed arm was poisoned. v2's exact projectors changed nothing because this bug
dominated everything.

**Why it hid so well:** the synthetic sanity check constructed the advected field with the *same*
wrong convention — self-consistent, so round trips passed. The honest detector was a
*statistical* diagnostic on real solver data: temporal-variance reduction. Lesson recorded:
**unit-convention bugs are invisible to self-consistent round trips; validate against an
independent statistical signature of the physics (here: a drift-free frame must have ~zero
temporal variance).**

**Fix (v3):** `c_px(t) = U * (t * dt_out) * (S / L)`. Verified: t-var reduction −247% -> +96%,
oracle 0.1332 -> **0.0009**.

### Bug 2: Cole–Hopf quadrature + the Nyquist-bin subtlety
v1 used trapezoid cumsum for the integral and central differences for the derivative: O(dx^2)
error, and inconsistent discretizations forward vs backward. v2's spectral version (FFT integral
`I_hat_k = u_hat_k / (i w_k)`, spectral derivative `d/dx <-> i w_k`) is *algebraically* exact —
but exposed a genuine subtlety:

**Discovery (Nyquist bin):** the discrete periodic integral of a signal with a *nonzero Nyquist
mode* requires that mode's coefficient to become **imaginary** (`u_N/(i w_N)`), but `irfft`
produces real sequences and silently drops the imaginary part. The round trip therefore loses
exactly the Nyquist mode (error = alternating +-1 pattern, std == max). The discrete periodic
integral is only well-defined on the zero-Nyquist band. Fix: zero the Nyquist bin in the forward
transform — for our solver, which dealiases at 2/3 Nyquist, that bin is already ~0, so the map
is exact with zero information loss. Verified: round trip 3e-4 (v1) -> **1.7e-6**, even on tanh
shocks (1.7e-6 dealiased).

### Bug 3: Gibbs ringing in fractional Fourier shifts
A Fourier phase-ramp shift is exact **only for band-limited fields**. Shock spectra decay like a
power law past the solver's dealias threshold, so v1's per-frame fractional shifts rang.
Fix: band-limit to the solver's own 2/3 band before shifting — zero information loss (the solver
never populated that band), exact shifts by the sampling theorem.

### Design flaw: learning a phase that is available in closed form
After the fixes, the Galilean oracle hit 0.0009 but the *learned* arm sat at 0.1406: reconstructing
near-vertical shocks requires ~0.02 px displacement precision, and a generic phase-regression head
stalls around 15 px RMS. But the drift is not an unknown: **U = mean(u0) is an exact conserved
invariant** (verified to 3.6e-07 across all samples and timesteps). Reading it in closed form from
u0 is exactly as fair as Reflection's deterministic sign canonicalization. v4 added the
"Galilean closed-form" arm: learned = oracle = 0.0009.

---

## Part 2 — The full new math

### Setup
Viscous Burgers on the periodic torus `x in [0, L)`, `L = 20`, `nu = 0.02`:

    u_t + u u_x = nu u_xx ,      u(., 0) = u0

Data: `{u0, u(., t_i)}` at output times `t_i = i * dt_out`, `dt_out = 0.15`, `i = 1..T`, `T = 20`.
Grid: `S = 512` points, `x_j = j L / S`. All maps below are bijections of the solution space;
canonicalization applies them left-to-right, evaluation applies the exact inverses right-to-left.

### Symmetry 1 — Galilean boost (co-moving frame), the workhorse
*Conserved invariant:* `U = (1/L) * ∫ u0 dx` (periodic flux-form solver conserves it exactly;
verified 3.6e-07).

*Lie symmetry:* if `u` solves Burgers, then for any boost `U_b`:
`u'(x, t) = u(x - U_b t, t) + U_b` also solves. Choosing `U_b = -U`:

    w(x, t) := u(x + U t, t) - U   =   v(x, t),

where `v` solves Burgers with initial data `v0 = u0 - U` — a **drift-free** field (zero mean).

*Discrete form:* `c_px(t_i) = U * (t_i * dt_out) * (S / L)` pixels;
`w_i = T_{-c}(u_i) - U` with `T_c f(x) = F^{-1}[e^{-i k c} f_hat](x)` applied on the
band-limited field (Symmetry 4's band). Inverse: `u_hat(x, t) = w(x - U t, t) + U`.

### Symmetry 2 — Reflection (discrete)
`u(x, t) -> -u(-x, t)` maps solutions to solutions (odd in both u and x; the PDE is invariant).
Grid: `v_j = -u_{(S-j) mod S}` (a roll+flip; exact involution, zero numerical error).
Canonicalization: flip so `sign(U) > 0`; decision read from u0 only.

### Symmetry 3 — Dilation (scaling), exact at fixed viscosity
    u(x, t) -> lam * u(lam x, lam^2 t)      preserves nu     (any lam > 0)

(derivation: substituting `u_hat(x,t) = A u(Bx, Ct)` into the PDE forces `A = B`, `C = B^2`.)
Canonicalization: `lam_i = clip(ref / RMS(w_i), 1, 1.25)` with `ref` = median RMS over the
**train split only**; `lam >= 1` keeps canonical times `tau = t / lam^2 <= T` (no horizon
extrapolation). Numerics: band-limited Fourier resampling (zero-padded spectrum, 16x fine grid).

### Symmetry 4 — Cole–Hopf / heat-semigroup representation (the linearization)
    phi = exp( -(1/(2 nu)) * Integral u dx )        maps Burgers -> heat equation phi_t = nu phi_xx
    inverse:  u = -2 nu (phi_x / phi)

In phi space the **exact solution operator is the heat semigroup**: `phi_hat_k(t) =
exp(-nu k^2 t) phi_hat_k(0)` — linear, mode-decoupled, diagonal in Fourier space. An FNO is a
spectral architecture, so this representation matches its ideal function class: the learned map
becomes close to a diagonal multiplier instead of nonlinear shock advection. (Operator-learning
literature uses precisely this route: Gin-Lusch-Brunton-Kutz arXiv:1911.02710; Xu-Guilleminot-
Tarokh, CMAME 444:118148, 2025.)

*Discrete exact pair (v2+):* on the **de-meaned** field `um = u - mean(u)` (a nonzero-mean
torus field has quasi-periodic log-phi — a sawtooth seam that must not enter the representation;
the DC is carried separately as an exact invariant, see leakage audit):

    I_hat_k = um_hat_k / (i w_k)   for k = 1..S/2 - 1,   I_hat_0 = I_hat_{S/2} = 0
    lp = -(2 nu / L) * (1/(2 nu)) I = -I / L,   centered
    inverse:  d/dx lp  <->  i w_k multiplication;   u = -2 nu d(lp)/dx + mean(u)

`w_k = 2 pi k / L`. rfft and irfft are exact inverses on the grid, and the `i w` factors cancel
algebraically, so the pair is machine-precision exact — with the Nyquist bin zeroed (see Bug 2).
Scaling: `lp` is O(1) by construction (`lp = -I/L`).

*Conjugated forms (when Cole–Hopf is applied before other maps):* on the de-meaned rep,
Galilean is a **pure shift** `lp'(x) = lp(x + U t)` (no quasi-periodic ramp — that was the v1
Gibbs trap); reflection is a pure flip; dilation is `lp'(x, tau) = (1/lam) lp(lam x, lam^2 tau)`.

### Composition and dynamic order selection
Reflection is pinned first (its sign is defined by sign U); the remaining three permute freely —
`3! = 6` candidate orders, each an exact composed symmetry map. The order is chosen by a tiny
validation-only screening run (15% budget); the TOP-2 merged arm re-screens its own orders.
Winner at full budget: `reflect -> galilean -> scale -> colehopf`.

### Evaluation protocol (fairness contract)
- Shared baseline: trained once on raw `(u0 -> futures)`, never modified.
- Symmetry parameters from **u0 only**: `U = mean(u0)` (exact invariant, 3.6e-07), flip sign,
  `lam` from canonical-u0 RMS + train median, Cole–Hopf DC = exact invariant.
- Oracle arms (ground-truth-derived parameters) are reported separately, labeled oracle.
- Learned arms: PhaseNet sees only u0 at test time; training targets follow the standard
  supervised convention.
- Identical splits / architectures / epochs / losses / relative-L2 metric for every arm.

---

## Part 3 — Leakage audit (no future-frame leakage)

| parameter | source | future access |
|---|---|---|
| drift `U` (closed-form arm) | `mean(u0)` — exact conserved invariant (verified 3.6e-07) | none |
| reflection sign | `sign(mean(u0))` | none |
| dilation `lam` | RMS of canonical u0 + train-split median | none |
| Cole–Hopf DC restoration | per-frame means — **provably constant = `mean(u0)`** (verified 3.6e-07 over all 64x21 frames) | none (invariant) |
| learned-phase arms at test | `PhaseNet(u0)` | none |
| oracle arms | derived from true frames **by definition** — labeled "oracle", reported separately | by design, labeled |
| training supervision | phase targets from ground truth (standard supervised learning) | train-time only, standard |
| order screening | validation split only | none |

One subtlety worth stating honestly: the Cole–Hopf DC restoration *reads* per-frame means from
the canonical arrays, but those values are provably the time-invariant `mean(u0)` (verified:
max deviation 3.576e-07 across every sample and timestep). A refactor could compute them from
frame 0 alone with identical output. No future-frame information reaches any test-time
prediction.

---

## Part 4 — Important discoveries

1. **Unit-convention bugs are invisible to self-consistent round trips.** The 6.67x Galilean
   over-shift passed every round-trip check for three notebook versions because the synthetic
   test shared the wrong convention. Detector that worked: a physics-signature diagnostic on real
   data (drift-free frame => temporal variance must collapse). Always pair exactness checks with
   an independent *statistical* signature.
2. **The oracle column is the diagnostic compass.** oracle >> baseline => the map's numerics are
   the bottleneck (fix math); oracle << baseline but learned >> oracle => the parameter learner
   is the bottleneck (fix training or use a closed-form invariant). This single rule drove the
   whole v1->v4 path.
3. **The Nyquist bin of the discrete periodic integral must be zeroed** — its coefficient would
   need to be imaginary, and irfft silently discards it, leaving exactly that mode as error.
4. **Nonzero-mean fields have quasi-periodic log-phi** (a sawtooth seam). Cole–Hopf must act on
   the de-meaned field with the DC carried as a separate invariant — which then makes the
   conjugated Galilean a *pure shift* instead of a ramp that Gibbs-rings.
5. **Discrete symmetries are free lunches; continuous ones have precision bills.** Reflection
   (zero error, zero parameters) beat everything at full budget until the continuous maps were
   fixed. And when a continuous parameter is an exact invariant of the input, use it in closed
   form — a regression head cannot reach the ~0.02px precision that shock re-alignment demands.
6. **Match the representation to the architecture.** The heat-semigroup representation turns the
   learned map into (nearly) a diagonal Fourier multiplier — the FNO's native language. This,
   not the baseline being weak, is why the final arm wins by 11x with the *same* network and
   training budget.
