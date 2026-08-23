# Iterative transmission expansion with impedance feedback

Reference for `pypsa.optimization.abstract.optimize_transmission_expansion_iteratively`.
It specifies the two outer schemes (`method="fixed_point"` and
`method="trust_region"`), the optional proximal term (`proximal=True`) and the
convergence criterion, in the notation of the SSSC capacity expansion model.

---

## 1. What is nonlinear

The capacity expansion problem is a linear program except for the Kirchhoff
voltage law (KVL). Expanding an AC branch lowers its impedance,

$$\hat{x}_\ell(F_\ell) \;=\; \hat{x}^0_\ell \frac{F^0_\ell}{F_\ell},$$

and the series compensation of an SSSC enters the same equation divided by the
branch capacity. Per independent cycle $c$ and snapshot $t$ the exact law is

$$\sum_{\ell \in \mathcal{L}} C_{\ell,c}\, g_{\ell,t}(F_\ell, f_{\ell,t}, \tilde{q}_{\ell,t}) = 0,
\qquad
g_{\ell,t} \;=\; \hat{x}_\ell(F_\ell)\, f_{\ell,t} \;-\; \frac{\tilde{q}_{\ell,t}}{F_\ell}.$$

Both terms of $g$ are proportional to $1/F_\ell$, so $g$ is a nonlinear
(bilinear-rational) function of the decision variables. The ohmic losses are
**not** part of this nonlinearity: their piecewise-linear envelope is exact in
$F_\ell$, because the conductor scales as $r_\ell = \rho_\ell / F_\ell$ and the
flow range as $\pm F_\ell$, so the segment slopes are capacity independent and
the offsets are linear in $F_\ell$ and carried as a term in the capacity
variable.

---

## 2. Notation

### Sets and indices

| Symbol | Meaning |
| --- | --- |
| $\ell \in \mathcal{L}$ | passive branches (`Line`, `LineX`, `Transformer`) |
| $\mathcal{L}^{\text{ext}} \subseteq \mathcal{L}$ | branches with extendable capacity |
| $\mathcal{L}^{\text{scale}} \subseteq \mathcal{L}^{\text{ext}}$ | branches whose impedance scales with the capacity: extendable AC-carrier branches without a standard type, plus extendable branches with a type (scaled through `num_parallel`) |
| $c \in \mathcal{C}$ | independent cycles of the passive network, one basis per sub-network |
| $t \in T$ | snapshots |
| $n$ | outer iteration counter |

### Data

| Symbol | Code | Meaning |
| --- | --- | --- |
| $C_{\ell,c} \in \{-1,0,1\}$ | `sub.C` | orientation of branch $\ell$ in cycle $c$ |
| $F^0_\ell$ | `s_nom` at entry | capacity the given impedance refers to |
| $\hat{x}^0_\ell$ | `x_pu_eff` at entry | per-unit reactance at $F^0_\ell$ |
| $w_\ell$ | `x_pu_eff` / `r_pu_eff` | impedance entering KVL: reactance in AC, resistance in DC sub-networks |
| $c_\ell$ | `capital_cost` | annualised capital cost per MVA of branch $\ell$ |
| $F^{\min}_\ell, F^{\max}_\ell$ | `s_nom_min`, `s_nom_max` | capacity bounds as given by the user |
| $\varepsilon_{\text{cost}}$ | `cost_threshold` | convergence tolerance on the system cost |
| $K$ | `cost_window` | number of consecutive cost changes that must undercut it |
| $\eta$ | `PROXIMAL_CALIBRATION` | share of $\varepsilon_{\text{cost}}$ charged for a typical capacity move |
| $\rho_{\min}, \rho_{\max}$ | `trust_region_bounds` | admissible trust region radii |
| $\varepsilon_{\text{tgt}}, \varepsilon_{\max}$ | `trust_region_tolerances` | linearisation error that is good enough / too large |
| $\sigma, \gamma$ | `trust_region_factors` | shrink and expand factor of the radius |
| $\tau$ | `sensitivity_tolerance` | loading below which a branch sensitivity is dropped |

### Iteration quantities

| Symbol | Code | Meaning |
| --- | --- | --- |
| $\bar{F}^n_\ell$ | `current_def`, `_s_nom_def` | **linearisation point**: the capacity the impedances of iteration $n$ are evaluated at |
| $\bar{x}^n_\ell = \hat{x}^0_\ell F^0_\ell / \bar{F}^n_\ell$ | `x_pu_eff` | reactance used in iteration $n$ |
| $F^{n,*}_\ell$ | `s_nom_opt` | capacity returned by the LP of iteration $n$ |
| $f^{n,*}_{\ell,t}$ | `p0` | branch flow returned by the LP |
| $\tilde{q}^{n,*}_{\ell,t}$ | `q_sssc` | SSSC control variable returned by the LP |
| $g^n_{\ell,t}$ | `cycle_terms` | cycle term of the new iterate, evaluated at **its own** capacities $F^{n,*}$ |
| $\alpha^n_{\ell,t}$ | `_kvl_capacity_sensitivity` | capacity sensitivity of the cycle term, truncated by $\tau$ |
| $u_\ell$ | `{Line,LineX}-s_nom_relative` | relative deviation of the capacity from the linearisation point |
| $V^n, \hat{V}^n$ | `violation`, `violation_rel` | absolute and relative residual of the exact KVL |
| $s^n$ | `step` | relative capacity change ("msq"), **diagnostic only** |
| $\hat{C}^n$ | `cost` | system cost of the iterate |
| $\rho^n$ | `radius` | trust region radius |
| $\Delta^n_\ell$ | — | trust region half width of branch $\ell$ |
| $M^n$ | — | capital cost of the capacity moved by the step |
| $\delta^n$ | `proximal` | weight of the proximal term |

---

## 3. The inner problem

Let $\mathrm{LP}(\bar{F}, \alpha, \Delta, \delta)$ be the linear program of the
capacity expansion model in which

* the branch impedances are those of $\bar{F}$,
* the KVL constraint is the one of section 5.2 with sensitivities $\alpha$,
  carried by the relative capacity deviations $u$ where $\alpha \neq 0$,
* the capacities are restricted to the box of half width $\Delta$,
* the objective carries the proximal term of section 6 with weight $\delta$.

Setting $\alpha = 0$, $\Delta = \infty$, $\delta = 0$ recovers the LP of the
plain fixed-point scheme, which is the one of the paper — the deviation
variables are then not created at all.

A capacity vector is a **solution of the nonlinear problem** if it reproduces
itself, $F^{n,*} = \bar{F}^n$, in which case the exact KVL holds, because the
frozen impedances then are the impedances of the reported capacities.

---

## 4. Scheme A — fixed point (`method="fixed_point"`)

The impedances are frozen at the previous iterate and the KVL stays zeroth
order in the capacity:

$$\sum_{\ell} C_{\ell,c}\left( \bar{x}^n_\ell f_{\ell,t} - \frac{\tilde{q}_{\ell,t}}{\bar{F}^n_\ell} \right) = 0,
\qquad \bar{F}^{n+1} = F^{n,*},$$

This is Algorithm 1 of the paper and remains the default.

---

## 5. Scheme B — linearised KVL + trust region (`method="trust_region"`)

### 5.1 First-order model of the cycle term

Write $a_\ell = \hat{x}^0_\ell F^0_\ell$, so that
$g_{\ell,t} = (a_\ell f_{\ell,t} - \tilde{q}_{\ell,t}) / F_\ell$ and

$$\left.\frac{\partial g_{\ell,t}}{\partial F_\ell}\right|_{\bar{F}^n}
= -\frac{a_\ell \bar{f}_{\ell,t} - \bar{q}_{\ell,t}}{\left(\bar{F}^n_\ell\right)^2}
= -\frac{g^{\,n-1}_{\ell,t}}{\bar{F}^n_\ell} \;=:\; -\alpha^n_{\ell,t}.$$

The sensitivity is thus the cycle term of the linearisation point divided by
its capacity — no extra evaluation is needed. Expanding to first order in
$(F, f, \tilde{q})$ around $(\bar{F}^n, \bar{f}, \bar{q})$ and simplifying,

$$\boxed{\;g_{\ell,t} \;\approx\; \bar{x}^n_\ell f_{\ell,t} - \frac{\tilde{q}_{\ell,t}}{\bar{F}^n_\ell} - \alpha^n_{\ell,t}\left(F_\ell - \bar{F}^n_\ell\right)\;}$$

Two properties follow, and both are used by the algorithm:

1. The model is **exact for $F_\ell = \bar{F}^n_\ell$** and any $(f, \tilde{q})$:
   the correction vanishes and the constraint collapses onto the fixed-point
   constraint. A converged iterate therefore satisfies the exact KVL.
2. The error is exactly

   $$g^{\text{exact}} - g^{\text{lin}} = \frac{F_\ell - \bar{F}^n_\ell}{\bar{F}^n_\ell}\left(g^{\,n-1}_{\ell,t} - g^{\text{exact}}_{\ell,t}\right),$$

   the product of the relative capacity change and the change of the term. It
   is second order, and it is what the trust region has to keep small.

### 5.2 The constraint as implemented

The correction $\alpha^n_{\ell,t}(F_\ell - \bar{F}^n_\ell)$ is **not** written on
the capacity itself but on the dimensionless relative deviation

$$u_\ell \;=\; \frac{F_\ell}{\bar{F}^n_\ell} - 1,
\qquad\text{defined by}\qquad
\frac{1}{\bar{F}^n_\ell} F_\ell - u_\ell \;=\; 1
\quad \forall\, \ell \in \mathcal{L}^{\text{ext}},$$

one extra column and one extra row per extendable branch. With
$\alpha^n_{\ell,t} = 0$ for $\ell \notin \mathcal{L}^{\text{scale}}$, the
constraint added per cycle $c$ and snapshot $t$ is then

$$\sum_{\ell} C_{\ell,c}\left( \bar{x}^n_\ell f_{\ell,t} - \frac{\tilde{q}_{\ell,t}}{\bar{F}^n_\ell} - \alpha^n_{\ell,t}\,\bar{F}^n_\ell\, u_\ell \right)
\;=\; 0 ,
\qquad \alpha^n_{\ell,t}\,\bar{F}^n_\ell = g^{\,n-1}_{\ell,t}.$$

The whole row is scaled by $10^4$ as in the original formulation. The
coefficient on $u_\ell$ is time dependent, so those rows carry two-dimensional
coefficients; the right-hand side, however, is a plain zero.

The first iteration has no linearisation point and is therefore a plain
fixed-point step, with neither the deviation variables nor the box.

### 5.2.1 Why the deviation and not the capacity

Writing the correction on $F_\ell$ directly is algebraically identical but
numerically unusable, for two reasons.

**The right-hand side is a cancelling sum.** It would be
$-\sum_\ell C_{\ell,c}\alpha^n_{\ell,t}\bar{F}^n_\ell = -\sum_\ell C_{\ell,c} g^{\,n-1}_{\ell,t}$,
i.e. exactly the KVL residual of the linearisation point. On the cycles and
snapshots the previous iterate already satisfies, that is a sum of terms of
order $10^{-3}$–$10^{0}$ cancelling down to round-off, and the solver is handed
right-hand sides of order $10^{-10}$ that drift upward as the linearisation
moves.

**The coefficients collapse.** The coefficient on $F_\ell$ is
$10^4 C_{\ell,c}\alpha^n_{\ell,t} = 10^4 C_{\ell,c} w_\ell f_{\ell,t} / \bar{F}^n_\ell$,
i.e. the flow coefficient of the same branch in the same row multiplied by its
loading $f_{\ell,t}/\bar{F}^n_\ell \in [0,1]$ and divided by nothing that brings
it back. A branch that is idle at a snapshot therefore contributes a
coefficient several orders of magnitude below every other entry of its row.
This is a spread *within* the row, so row scaling cannot remove it, and column
scaling cannot either: the same column carries the capital cost, the flow
limits and the volume limit, where the MW scale is the correct one.

The substitution $F_\ell = \bar{F}^n_\ell (1 + u_\ell)$ is precisely the missing
column scaling, applied to a column that is used by the voltage law alone. It
multiplies every sensitivity coefficient by $\bar{F}^n_\ell$, which puts it on
the scale of the flow coefficients of its own row, and it absorbs the constant
of the linearisation, which leaves the right-hand side at zero. Since
$F_\ell \ge 0$, the bound $u_\ell \ge -1$ is exact and needs no assumption on
the step size; the box of section 5.3 stays on $F_\ell$.

### 5.2.2 Dropping negligible sensitivities

The rescaling fixes the *scale* of the sensitivity coefficients but not their
*spread*: the coefficient of $u_\ell$ is $10^4 C_{\ell,c} g^{\,n-1}_{\ell,t}$
and still vanishes with the flow of the branch. The flows come from a barrier
solve without crossover, so an idle branch is never returned at exactly zero
but at $10^{-9}$ MW or so, which is enough to keep the entry in the matrix.

Sensitivities that are negligible against the voltage drop their **own** branch
causes at its rated capacity are therefore set to zero before the model is
built:

$$\left| g^{\,n-1}_{\ell,t} \right| \;\le\; \tau\, \left| w_\ell \bar{F}^n_\ell \right|
\quad\Longrightarrow\quad \alpha^n_{\ell,t} \leftarrow 0 ,$$

with $\tau =$ `sensitivity_tolerance`, default $10^{-6}$. The criterion is a
loading threshold: a branch loaded below $\tau$ of its rating at a snapshot has
no influence on the voltage law of that snapshot. The surviving coefficients
are bounded below by $\tau \cdot 10^4 |w_\ell \bar{F}^n_\ell|$, i.e. by $\tau$
times a quantity the impedance scaling leaves invariant under the iteration, so
the range of the constraint matrix no longer degrades from one iteration to the
next. The dropped terms perturb their row by less than $\tau$ of its rated
voltage drop — three orders of magnitude below $\varepsilon_{\text{tgt}}$, the
linearisation error the iteration considers good. `sensitivity_tolerance=0`
keeps every sensitivity.

The linearisation error of section 5.3 is measured on the untruncated cycle
terms $g^n$, so the truncation cannot hide itself from the acceptance test.

### 5.3 Trust region

Half width and box, with the radius relative to the linearisation point and
floored by the initial capacity so it cannot collapse for a branch shrinking
towards zero:

$$\Delta^n_\ell = \rho^n \max\left(\bar{F}^n_\ell,\, F^0_\ell\right),
\qquad
\max\left(F^{\min}_\ell,\, \bar{F}^n_\ell - \Delta^n_\ell\right) \le F_\ell \le \min\left(F^{\max}_\ell,\, \bar{F}^n_\ell + \Delta^n_\ell\right).$$

**Linearisation error.** The linear model predicts a vanishing residual, so
whatever residual the new iterate leaves is the model error. It is evaluated at
the capacities of the new iterate — i.e. the impedances are first set to
$F^{n,*}$ — and normalised by the voltage drop the branches of the cycle cause
at their rated capacity. That scale is stable under the iteration: for every
branch whose impedance scales with the capacity it is exactly invariant, since
$w_\ell F_\ell = \hat{x}^0_\ell F^0_\ell$, and for the remaining ones both
factors are fixed by the input data:

$$V^n = \sum_{c,t}\left| \sum_\ell C_{\ell,c}\, g^n_{\ell,t} \right|,
\qquad
S = |T| \sum_{c} \sum_{\ell} \left|C_{\ell,c}\right| w_\ell F_\ell,
\qquad
\hat{V}^n = V^n / S .$$

**Binding.** The region restricts the step if the branches sitting at its
boundary carry a relevant share of the moved capital cost:

$$\text{binding}^n \;=\;
\left[\;
\frac{\sum_{\ell \,:\, |F^{n,*}_\ell - \bar{F}^n_\ell| \ge 0.9\,\Delta^n_\ell} c_\ell \left|F^{n,*}_\ell - \bar{F}^n_\ell\right|}
{\sum_{\ell} c_\ell \left|F^{n,*}_\ell - \bar{F}^n_\ell\right|} \;\ge\; 0.1
\;\right]$$

The cost weighting matters: in the maximum norm a single small branch swinging
between degenerate optima already fills its own width and would report a
restriction that does not exist.

**Acceptance and radius**, evaluated in this order and only while a
linearisation is active and the iteration has not converged:

| condition | action |
| --- | --- |
| $\hat{V}^n > \varepsilon_{\max}$ | reject the step, $\rho \leftarrow \max(\sigma\rho, \rho_{\min})$ |
| else if $s^n \ge s^{n-1}$ | accept, $\rho \leftarrow \max(\sigma\rho, \rho_{\min})$ — no progress, the model is trusted over too wide a range |
| else if $\hat{V}^n \le \varepsilon_{\text{tgt}}$ and binding | accept, $\rho \leftarrow \min(\gamma\rho, \rho_{\max})$ |
| otherwise | accept, radius unchanged |

A rejected step leaves $\bar{F}$ untouched. If the radius has to be shrunk
below $\rho_{\min}$, the iteration stops and reports the last accepted iterate.

### 5.4 Fallback

The linearised LP can be infeasible where the frozen impedances require more
capacity than the box admits. Shrinking would make this worse, so the next
iteration is taken as an unrestricted fixed-point step, which is always
feasible, and the radius is shrunk for the iteration after it. The same
fallback is used if a solve reports success but returns capacities below their
own lower bounds, which is how a numerically failed solve shows up — such a
point must not become a linearisation point.

---

## 6. Proximal term (`proximal=True`)

### 6.1 Formulation

Large meshed networks have many capacity vectors of equal cost. The iterates
then keep exchanging capacity between branches without any cost change, which
leaves $s^n$ bouncing and the capacity vector unsettled. The proximal term
penalises *moving* away from the previous iterate, in units of capital cost:

$$\min \; \left(\text{system cost}\right) \;+\; \delta^n \sum_{\ell \in \mathcal{L}^{\text{ext}}} c_\ell \left|F_\ell - \bar{F}^n_\ell\right|$$

modelled with one non-negative deviation variable per extendable branch, which
the penalty drives onto its bound:

$$d_\ell \ge F_\ell - \bar{F}^n_\ell, \qquad d_\ell \ge \bar{F}^n_\ell - F_\ell, \qquad d_\ell \ge 0 .$$

This adds $|\mathcal{L}^{\text{ext}}|$ variables and $2|\mathcal{L}^{\text{ext}}|$
rows, none of them time dependent — negligible next to a model with snapshots.
The term is zero at a fixed point and is subtracted again from the reported
cost, so it never enters the results.

### 6.2 Calibration

The weight is not given but derived from the convergence criterion. With the
capital cost of the capacity moved by the previous step,

$$M^n = \sum_{\ell \in \mathcal{L}^{\text{ext}}} c_\ell \left|F^{n,*}_\ell - \bar{F}^n_\ell\right|,$$

the weight of the next iteration is

$$\delta^{n+1} = \operatorname{clip}\!\left(\frac{\eta\, \varepsilon_{\text{cost}} \left|\hat{C}^n\right|}{M^n},\; 10^{-6},\; 10^{-1}\right).$$

A move of the size of the previous one is thus charged $\eta$ times the cost
difference the iteration considers converged: only moves worth more than that
are taken. Because $\delta$ is recalibrated in every iteration it **grows as
the moves get smaller**, which anneals the iteration — a small $\eta$ only
delays the damping by a few iterations instead of removing it. Measured on the
SciGRID case: $\eta = 1$ damps hardest but shifts the reported cost by
$\sim 10^{-3}$ relative, $\eta = 0.1$ damps nearly as well at $\sim 10^{-5}$,
$\eta = 0.01$ never anneals into effect. Hence the default $\eta = 0.1$.

---

## 7. Convergence criterion

**System cost of an iterate.** Let $z^{n,*}$ be the objective value of the LP.
The cost of the already installed capacity is subtracted as determined in the
first iteration, and the proximal penalty is removed:

$$\hat{C}^n = z^{n,*} - K^{1} - \delta^n M^n,
\qquad K^{1} = \sum_{c,\,\text{ext}} \left(\text{capital cost}\right) \times \left(\text{installed capacity}\right) \Big|_{n=1}.$$

Freezing $K^1$ is necessary: for branches with a standard type PyPSA derives
`s_nom` from `num_parallel`, so the objective constant that `n.objective`
subtracts drifts with the linearisation capacity and is not comparable between
iterations.

**Criterion.** Over the accepted iterates,

$$\left|\hat{C}^{\,n-k} - \hat{C}^{\,n-k-1}\right| \Big/ \left|\hat{C}^n\right| \;\le\; \varepsilon_{\text{cost}}
\qquad \text{for } k = 0, \dots, K-1 .$$

**Why not the capacity change.** $s^n = \lVert F^{n,*} - \bar{F}^n \rVert_2 / \lVert F^0 \rVert_2$
is a step size, not a measure of convergence. It is not invariant under the
exchange of degenerate alternative optima, so it keeps bouncing long after the
solution has settled, and a step can undercut a threshold by accident while the
cost is still drifting. On the SciGRID case both failure modes occur: the fixed
point passes $s^n < 10^{-2}$ at iteration 10 while its cost still moves by
$4.5\cdot10^{-4}$ per iteration, and the trust region has a converged cost at
iteration 7 with $s^n \approx 5\cdot10^{-2}$. The quantity is still reported as
`step` because it is a useful diagnostic of degeneracy.

---

## 8. Measured behaviour on the German system

SciGRID Germany — 585 buses, 852 AC lines, 96 transformers, 6 snapshots — set up
as a brownfield expansion in which renewable generation, storage, the AC lines
and, in the SSSC case, the series compensation are co-optimised. Solved with
Gurobi, `transmission_losses=2`, `cost_threshold=1e-5`.

![System cost per iteration for both schemes on the German system with SSSC](img/iterative-expansion-cost-sssc.svg)

Both schemes start from the same first LP, whose impedances still belong to the
unexpanded network. From there the fixed point spends 34 iterations creeping
along a band it never leaves, because every impedance update redistributes the
flows again; the trust region anticipates that redistribution, overshoots once
at iteration 2 and has settled from iteration 5 on. The two plans differ by
0.085 % in cost — and by 8 % in the transmission expansion they call for.

| | iterations | system cost | expansion | cost change at stop | capacity change at stop |
| --- | ---: | ---: | ---: | ---: | ---: |
| **without SSSC** — fixed point | 28 | 3.99115e6 | 100.5 GVA | 2.9e-6 | 2.7e-3 |
| **without SSSC** — trust region | **7** | 3.96360e6 | 77.1 GVA | 2.4e-6 | 4.9e-2 |
| **with SSSC** — fixed point | 34 | 3.86067e6 | 71.1 GVA | 9.6e-7 | 1.5e-3 |
| **with SSSC** — trust region | **7** | 3.85739e6 | 65.3 GVA | 1.3e-6 | 3.5e-2 |

Two observations concern the results of the model rather than its numerics.
First, the scheme changes the reported transmission expansion by 8 % (with SSSC)
to 23 % (without) — the same order as the effect of the SSSCs themselves, so the
choice of scheme is not an implementation detail of the solution report. Second,
the capacity change at the stopping point is one to two orders of magnitude
*larger* for the converged trust region than for the fixed point, which is
precisely why it cannot serve as the criterion.

<details>
<summary>Data of the figure</summary>

| iteration | fixed point | trust region |
| ---: | ---: | ---: |
| 1 | 3.85824 | 3.85824 |
| 2 | 3.85480 | 3.86105 |
| 3 | 3.86003 | 3.85783 |
| 4 | 3.86051 | 3.85741 |
| 5 | 3.86055 | 3.85739 |
| 6 | 3.86052 | 3.85739 |
| 7 | 3.86064 | 3.85739 |
| 8 | 3.86059 | converged |
| 9 | 3.86063 |  |
| 10 | 3.86059 |  |
| 11 | 3.86058 |  |
| 12 | 3.86055 |  |
| 13 | 3.86059 |  |
| 14 | 3.86053 |  |
| 15 | 3.86058 |  |
| 16 | 3.86053 |  |
| 17 | 3.86058 |  |
| 18 | 3.86053 |  |
| 19 | 3.86061 |  |
| 20 | 3.86058 |  |
| 21 | 3.86064 |  |
| 22 | 3.86060 |  |
| 23 | 3.86065 |  |
| 24 | 3.86060 |  |
| 25 | 3.86065 |  |
| 26 | 3.86062 |  |
| 27 | 3.86066 |  |
| 28 | 3.86062 |  |
| 29 | 3.86066 |  |
| 30 | 3.86062 |  |
| 31 | 3.86066 |  |
| 32 | 3.86064 |  |
| 33 | 3.86066 |  |
| 34 | 3.86067 |  |

</details>

---

## 9. Algorithm

```text
input  F⁰, initial impedances, ε_cost, K, η, ρ⁰, σ, γ, ε_tgt, ε_max, n_min, n_max
init   F̄ ← F⁰,  n ← 1,  ρ ← ρ⁰,  δ ← 0,  ref ← ∅,  plain ← false,  costs ← []

while n ≤ n_max:
    set branch impedances from F̄                      # x = x̂⁰F⁰/F̄, typed: num_parallel
    if trust_region and ref ≠ ∅ and not plain:
        α ← ref / F̄        on ℒ_scale                 # sensitivities, eq. (5.1)
        α ← 0 where |ref| ≤ τ·|w·F̄|                   # drop the idle branches, 5.2.2
        apply trust region box of radius ρ around F̄
    else:
        α ← 0;  no box                                 # iteration 1, fixed point, fallback
    plain ← false
    anchor ← F̄ if n > 1 else ∅                        # proximal anchor
    solve LP(F̄, α, ρ, δ; anchor)

    if infeasible or failed:
        if α ≠ 0: ρ ← max(σρ, ρ_min); plain ← true; n ← n+1; continue
        else: raise
    F* ← optimal capacities
    if F* below its own lower bounds:                  # numerically failed solve
        warn; if α ≠ 0: ρ ← max(σρ, ρ_min); plain ← true; n ← n+1; continue

    Ĉ ← z* − K¹ − δ·M(F*, F̄)                          # system cost, section 7
    s ← ‖F* − F̄‖ / ‖F⁰‖                               # diagnostic
    set branch impedances from F*                      # evaluate the exact law at F*
    g, V, V̂ ← cycle terms and KVL residual             # section 5.3

    converged ← cost stationary over K iterations and n ≥ n_min
    binding   ← cost-weighted share at the boundary ≥ 0.1
    if α ≠ 0 and not converged:  accept / reject and update ρ    # table in 5.3

    if rejected:
        if ρ ≤ ρ_min: stop
        n ← n+1; continue
    if converged:
        F̄ ← clip(F*);  ref ← g;  break
    F̄ ← clip(F*) if trust_region else F̄ + ω(F* − F̄)   # relaxation only for fixed point
    ref ← g;  δ ← calibrate(M, Ĉ);  costs ← costs + [Ĉ];  n ← n+1

final solve:
    set branch impedances from F̄
    if trust_region and converged:  keep α = ref/F̄ and the box   # exact there
    else:                           plain formulation            # Algorithm 1 of the paper
```

The final solve deserves a note. For the fixed-point scheme it is the
relaxation with frozen impedances, whose optimum can undercut the converged one
by expanding capacity without paying for the flow redistribution the lower
impedance causes. For a converged trust region the linearisation is exact at
that point, so keeping it reports the solution the iteration actually converged
to, with capacities and flows consistent.

---

## 10. Parameters and defaults

| Parameter | Default | Meaning |
| --- | --- | --- |
| `method` | `"fixed_point"` | scheme, see sections 4 and 5 |
| `cost_threshold` $\varepsilon_{\text{cost}}$ | `1e-5` | convergence tolerance on the system cost |
| `cost_window` $K$ | `3` | consecutive cost changes that must undercut it |
| `proximal` | `False` | switch of the proximal term |
| `PROXIMAL_CALIBRATION` $\eta$ | `0.1` | module constant, see 6.2 |
| `trust_region_initial` $\rho^1$ | `0.5` | initial radius, relative to $\max(\bar{F}, F^0)$ |
| `trust_region_bounds` | `(1e-3, 4.0)` | $(\rho_{\min}, \rho_{\max})$ |
| `trust_region_tolerances` | `(1e-3, 1e-1)` | $(\varepsilon_{\text{tgt}}, \varepsilon_{\max})$ |
| `trust_region_factors` | `(0.5, 2.0)` | $(\sigma, \gamma)$ |
| `sensitivity_tolerance` $\tau$ | `1e-6` | loading below which a branch sensitivity is dropped, see 5.2.2; `0` keeps all |
| `min_iterations`, `max_iterations` | `1`, `100` | iteration bounds |
| `msq_threshold` | — | ignored, kept for compatibility |

## 11. Diagnostics

`n.iteration_log` holds one row per iteration with

`status`, `cost` $\hat{C}^n$, `cost_change`, `step` $s^n$, `violation` $V^n$,
`violation_rel` $\hat{V}^n$, `radius` $\rho^n$, `binding`, `proximal` $\delta^n$,
`accepted`.

It lives on the network object only and is not written to file, so a run that
is interrupted loses it. The quantities that drive the radius are therefore
also written to the log at every iteration: `cost_change`, `step`,
`violation_rel` and, while a linearisation is active, `radius`, whether the
region was binding, and whether the step was accepted. That line is enough to
replay the decision table of section 5.3.

Every linearised iteration additionally logs how many of the branch
sensitivities were dropped by $\tau$ (section 5.2.2). A share close to one is
normal — most branches are idle at most snapshots — but a share close to zero
together with a `Model contains large matrix coefficient range` warning from the
solver means $\tau$ is too small for the network at hand.

Warnings are raised when a solve returns capacities below their own bounds,
when the radius reaches $\rho_{\min}$ without resolving the linearisation error,
and when the capacities converge against a binding trust region — the last one
means the reported point is consistent but may not be a local optimum.
