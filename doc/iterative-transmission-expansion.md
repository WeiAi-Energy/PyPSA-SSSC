# Iterative transmission expansion with impedance feedback

Reference for `pypsa.optimization.abstract.optimize_transmission_expansion_iteratively`.
It specifies the two outer schemes (`scheme="fixed_point"` and `scheme="slp"`),
the proximal term that bounds the step of the second one and the convergence
criterion, in the notation of the SSSC capacity expansion model. Equation numbers in parentheses
refer to that paper. All quantities of the voltage law are per unit; the
superscript $\mathrm{pu}$ is dropped throughout.

---

## 1. What is nonlinear

The capacity expansion problem is a linear program except for the Kirchhoff
voltage law (KVL). Expanding an AC branch lowers its impedance (9),

$$\hat{x}_\ell(F_\ell) \;=\; \hat{x}^0_\ell \frac{F^0_\ell}{F_\ell},$$

and the series compensation of an SSSC enters the same equation divided by the
branch capacity. Per independent cycle $c$ and snapshot $t$ the exact law (7) is

$$\sum_{\ell \in \mathcal{L}} C_{\ell,c}\, g_{\ell,t} = 0,
\qquad
g_{\ell,t} \;=\; \hat{x}_\ell(F_\ell)\, f_{\ell,t} \;-\; \frac{\tilde{q}_{\mathrm{SSSC},\ell,t}}{F_\ell}.$$

Both terms of the branch term $g_{\ell,t}$ are proportional to $1/F_\ell$, so
$g_{\ell,t}$ is a nonlinear (bilinear-rational) function of the decision
variables, and both are covered by a single capacity sensitivity in section 5.1.

The ohmic losses are **not** part of this nonlinearity. Unlike Algorithm 1 of
the paper, which recomputes the envelope coefficients $\alpha_{\ell,m}$,
$\beta_{\ell,m}$ of (1r)/(10c) from $\mathbf{R}^{n,\mathrm{fix}}$ in every
iteration, the implementation writes the envelope exactly in $F_\ell$: with
$r_\ell = r^0_\ell F^0_\ell / F_\ell$ and the flow range $\pm F_\ell$, the
segment slopes are capacity independent and the offsets are linear in $F_\ell$,
so the offset is carried as a term in the capacity variable. Lines 3–4 of
Algorithm 1 are therefore needed for $\hat{\mathbf{X}}$ only, and the outer
iteration exists for the voltage law alone.

---

## 2. Notation

### Sets and indices

| Symbol | Meaning |
| --- | --- |
| $\ell \in \mathcal{L}$ | passive branches (`Line`, `LineX`, `Transformer`) |
| $\mathcal{L}^{\mathrm{ext}} \subseteq \mathcal{L}$ | branches with extendable capacity |
| $\mathcal{L}^{\mathrm{scale}} \subseteq \mathcal{L}^{\mathrm{ext}}$ | branches whose impedance scales with the capacity: extendable AC-carrier branches without a standard type, plus extendable branches with a type (scaled through `num_parallel`) |
| $c \in \mathcal{C}$ | independent cycles of the passive network, one basis per sub-network |
| $t \in T$ | snapshots |
| $n$ | outer iteration counter |

### Data

| Symbol | Code | Meaning |
| --- | --- | --- |
| $C_{\ell,c} \in \{-1,0,1\}$ | `sub.C` | orientation of branch $\ell$ in cycle $c$ (3) |
| $F^0_\ell$ | `s_nom` at entry | capacity the given impedance refers to |
| $\hat{x}^0_\ell$ | `x_pu_eff` at entry | natural reactance at $F^0_\ell$; in DC sub-networks the resistance $r^0_\ell$ takes its place throughout |
| $c_\ell$ | `capital_cost` | annualised capacity cost of branch $\ell$ (1a) |
| $F^{\min}_\ell, F^{\max}_\ell$ | `s_nom_min`, `s_nom_max` | capacity bounds as given by the user (1f) |
| $\varepsilon$ | `cost_threshold` | convergence tolerance, here on the system cost rather than on the capacities as in Algorithm 1 |
| $\delta$ | `proximal_weight` | weight of the proximal term; the initial weight where `proximal_adaptive` raises it |
| $\tau$ | `sensitivity_tolerance` | loading below which a branch sensitivity is dropped |

### Iteration quantities

| Symbol | Code | Meaning |
| --- | --- | --- |
| $F^{n,\mathrm{fix}}_\ell$ | `current_def`, `_s_nom_def` | **linearisation point**: the capacity the impedances of iteration $n$ are evaluated at |
| $\hat{x}^{n,\mathrm{fix}}_\ell = \hat{x}^0_\ell F^0_\ell / F^{n,\mathrm{fix}}_\ell$ | `x_pu_eff` | natural reactance used in iteration $n$ |
| $F^{n,*}_\ell$ | `s_nom_opt` | capacity returned by the LP of iteration $n$ |
| $f^{n,*}_{\ell,t}$ | `p0` | branch flow returned by the LP |
| $\tilde{q}^{\,n,*}_{\mathrm{SSSC},\ell,t}$ | `q_sssc` | SSSC control variable returned by the LP |
| $g^{n,*}_{\ell,t}$ | `cycle_terms` | branch term of the new iterate, evaluated at **its own** capacities $F^{n,*}$ |
| $g^{n,\mathrm{fix}}_{\ell,t}$ | `reference_terms` | branch term at the linearisation point, $= g^{n-1,*}_{\ell,t}$ |
| $g^{n,\mathrm{fix}}_{\ell,t} / F^{n,\mathrm{fix}}_\ell$ | `_kvl_capacity_sensitivity` | the derivative itself, truncated by $\tau$; the constraint multiplies it back by $F^{n,\mathrm{fix}}_\ell$ |
| $u_\ell$ | `{Line,LineX}-s_nom_relative` | relative deviation of the capacity from the linearisation point |
| $V^n, \hat{V}^n$ | `violation`, `violation_rel` | absolute and relative residual of the exact KVL |
| $\mathrm{step}^n$ | `step` | relative capacity change |
| $obj^n$ | `cost` | system cost of the iterate |

---

## 3. The inner problem

Let $\mathrm{LP}(F^{\mathrm{fix}}, g^{\mathrm{fix}})$ be problem (10) in which

* the branch impedances are those of $F^{\mathrm{fix}}$,
* the KVL constraint is the one of section 5.2 with the sensitivities carried by
  $g^{\mathrm{fix}}$ through the relative capacity deviations $u$,
* the objective carries the proximal term of section 6 anchored at
  $F^{\mathrm{fix}}$.

Setting $g^{\mathrm{fix}} = 0$ and $\delta = 0$ recovers (10d) itself, the LP of
the plain fixed-point scheme — the deviation variables are then not created at
all.

A capacity vector is a **solution of the nonlinear problem** if it reproduces
itself, $F^{n,*} = F^{n,\mathrm{fix}}$, in which case the exact KVL holds,
because the frozen impedances then are the impedances of the reported
capacities.

---

## 4. Scheme A — fixed point (`scheme="fixed_point"`)

The impedances are frozen at the previous iterate and the KVL stays zeroth
order in the capacity, which is (10d):

$$\sum_{\ell} C_{\ell,c}\left( \hat{x}^{n,\mathrm{fix}}_\ell f_{\ell,t} - \frac{\tilde{q}_{\mathrm{SSSC},\ell,t}}{F^{n,\mathrm{fix}}_\ell} \right) = 0,
\qquad F^{n+1,\mathrm{fix}} = F^{n,*}.$$

This is Algorithm 1 of the paper. It is kept for reference and comparison; the
default is scheme B. There is no linearisation error for a step control to keep
small, so the proximal term is inactive under this scheme and the step is the
bare fixed point.

---

## 5. Scheme B — linearised KVL (`scheme="slp"`, default)

### 5.1 First-order model of the branch term

With $g_{\ell,t} = \left(\hat{x}^0_\ell F^0_\ell f_{\ell,t} - \tilde{q}_{\mathrm{SSSC},\ell,t}\right) / F_\ell$,

$$\left.\frac{\partial g_{\ell,t}}{\partial F_\ell}\right|_{F^{n,\mathrm{fix}}}
= -\frac{\hat{x}^0_\ell F^0_\ell f^{n-1,*}_{\ell,t} - \tilde{q}^{\,n-1,*}_{\mathrm{SSSC},\ell,t}}{\left(F^{n,\mathrm{fix}}_\ell\right)^2}
= -\frac{g^{n,\mathrm{fix}}_{\ell,t}}{F^{n,\mathrm{fix}}_\ell}.$$

Because the natural reactance term and the SSSC term are **both** homogeneous of
degree $-1$ in $F_\ell$, this single sensitivity is the derivative of the whole
term; no separate treatment of the compensation is needed. It is the branch term
of the linearisation point divided by its capacity, so no extra evaluation is
needed either. Expanding to first order in
$(F_\ell, f_{\ell,t}, \tilde{q}_{\mathrm{SSSC},\ell,t})$ around
$(F^{n,\mathrm{fix}}_\ell, f^{n-1,*}_{\ell,t}, \tilde{q}^{\,n-1,*}_{\mathrm{SSSC},\ell,t})$,
the constant $g^{n,\mathrm{fix}}_{\ell,t}$ and the expansion of the two terms
that are linear at fixed capacity collapse, and what is left is

$$\boxed{\;g_{\ell,t} \;\approx\; \hat{x}^{n,\mathrm{fix}}_\ell f_{\ell,t} - \frac{\tilde{q}_{\mathrm{SSSC},\ell,t}}{F^{n,\mathrm{fix}}_\ell} - \frac{g^{n,\mathrm{fix}}_{\ell,t}}{F^{n,\mathrm{fix}}_\ell}\left(F_\ell - F^{n,\mathrm{fix}}_\ell\right)\;}$$

The first two terms are those of (10d); the third is the impedance feedback the
fixed-point scheme leaves to the outer iteration.

The constant of the expansion is thus not dropped but absorbed. Writing the
model purely in the increments
$(f - f^{n-1,*},\, \tilde{q} - \tilde{q}^{\,n-1,*},\, F - F^{n,\mathrm{fix}})$
instead would leave out $\sum_\ell C_{\ell,c}\, g^{n,\mathrm{fix}}_{\ell,t}$,
the exact KVL residual of the linearisation point, which is the quantity logged
as $V^n$ and is not zero — the previous LP enforced the voltage law at
$F^{n-1,\mathrm{fix}}$, not at $F^{n,\mathrm{fix}} = F^{n-1,*}$. Such a model
would admit the previous iterate as a feasible point of every cycle and would
stop enforcing the voltage law at the fixed point.

Two properties follow, and both are used by the algorithm:

1. The model is **exact for $F_\ell = F^{n,\mathrm{fix}}_\ell$** and any
   $(f, \tilde{q})$: the correction vanishes and the constraint collapses onto
   (10d). A converged iterate therefore satisfies the exact KVL, which is what
   lets the run be reported from that iterate (section 5.4).
2. The error is exactly

   $$g^{\mathrm{exact}}_{\ell,t} - g^{\mathrm{lin}}_{\ell,t} = u_\ell\left(g^{n,\mathrm{fix}}_{\ell,t} - g^{\mathrm{exact}}_{\ell,t}\right),
   \qquad u_\ell = \frac{F_\ell}{F^{n,\mathrm{fix}}_\ell} - 1,$$

   the product of the relative capacity change and the change of the term. It is
   second order, and it is what the proximal term has to keep small. At frozen
   flows it specialises to $g^{n,\mathrm{fix}}_{\ell,t}\, u_\ell^2/(1 + u_\ell)$.

### 5.2 The constraint as implemented

The correction is **not** written on the capacity itself but on the
dimensionless relative deviation

$$u_\ell \;=\; \frac{F_\ell}{F^{n,\mathrm{fix}}_\ell} - 1,
\qquad\text{defined by}\qquad
\frac{1}{F^{n,\mathrm{fix}}_\ell} F_\ell - u_\ell \;=\; 1
\quad \forall\, \ell \in \mathcal{L}^{\mathrm{ext}},$$

one extra column and one extra row per extendable branch of the components that
carry a sensitivity. With $g^{n,\mathrm{fix}}_{\ell,t} = 0$ for
$\ell \notin \mathcal{L}^{\mathrm{scale}}$, the constraint that replaces (10d)
per cycle $c$ and snapshot $t$ is

$$\sum_{\ell} C_{\ell,c}\left( \hat{x}^{n,\mathrm{fix}}_\ell f_{\ell,t} - \frac{\tilde{q}_{\mathrm{SSSC},\ell,t}}{F^{n,\mathrm{fix}}_\ell} - g^{n,\mathrm{fix}}_{\ell,t}\, u_\ell \right)
\;=\; 0 .$$

The coefficient of $u_\ell$ is the branch term of the linearisation point
itself: the factor $F^{n,\mathrm{fix}}_\ell$ of the deviation cancels the one in
the denominator of the sensitivity. The whole row is scaled by $10^4$ as in the
original formulation. That coefficient is time dependent, so those rows carry
two-dimensional coefficients; the right-hand side, however, is a plain zero.

The first iteration has no linearisation point and no previous iterate to be
anchored at, so it is a plain fixed-point step, with neither the deviation
variables nor the penalty.

### 5.2.1 Why the deviation and not the capacity

Writing the correction on $F_\ell$ directly is algebraically identical but
numerically unusable, for two reasons.

**The right-hand side is a cancelling sum.** It would be
$-\sum_\ell C_{\ell,c}\, g^{n,\mathrm{fix}}_{\ell,t}$, i.e. exactly the KVL
residual of the linearisation point discussed in section 5.1. On the cycles and
snapshots the previous iterate already satisfies, that is a sum of terms of
order $10^{-3}$–$10^{0}$ cancelling down to round-off, and the solver is handed
right-hand sides of order $10^{-10}$ that drift upward as the linearisation
moves.

**The coefficients collapse.** The coefficient on $F_\ell$ is
$10^4 C_{\ell,c}\, g^{n,\mathrm{fix}}_{\ell,t} / F^{n,\mathrm{fix}}_\ell$, which
for a branch without compensation is the flow coefficient of the same branch in
the same row multiplied by its loading
$f^{n-1,*}_{\ell,t}/F^{n,\mathrm{fix}}_\ell \in [0,1]$ and divided by nothing
that brings it back. A branch that is idle at a snapshot therefore contributes a
coefficient several orders of magnitude below every other entry of its row. This
is a spread *within* the row, so row scaling cannot remove it, and column
scaling cannot either: the same column carries the capacity cost of (1a), the
thermal limits (1s) and the volume limit (1h), where the MW scale is the correct
one.

The substitution $F_\ell = F^{n,\mathrm{fix}}_\ell (1 + u_\ell)$ is precisely
the missing column scaling, applied to a column that is used by the voltage law
alone. It multiplies every sensitivity coefficient by $F^{n,\mathrm{fix}}_\ell$,
which puts it on the scale of the flow coefficients of its own row, and it
absorbs the constant of the linearisation, which leaves the right-hand side at
zero. Since $F_\ell \ge 0$, the bound $u_\ell \ge -1$ is exact and needs no
assumption on the step size.

### 5.2.2 Dropping negligible sensitivities

The rescaling fixes the *scale* of the sensitivity coefficients but not their
*spread*: the coefficient of $u_\ell$ is
$10^4 C_{\ell,c}\, g^{n,\mathrm{fix}}_{\ell,t}$ and still vanishes with the flow
of the branch. The flows come from a barrier solve without crossover, so an idle
branch is never returned at exactly zero but at $10^{-9}$ MW or so, which is
enough to keep the entry in the matrix.

Sensitivities that are negligible against the voltage drop their **own** branch
causes at its rated capacity are therefore set to zero before the model is
built:

$$\left| g^{n,\mathrm{fix}}_{\ell,t} \right| \;\le\; \tau\, \hat{x}^{n,\mathrm{fix}}_\ell F^{n,\mathrm{fix}}_\ell
\quad\Longrightarrow\quad g^{n,\mathrm{fix}}_{\ell,t} \leftarrow 0 ,$$

with $\tau =$ `sensitivity_tolerance`, default $10^{-5}$. On an uncompensated
branch the criterion is a loading threshold: a branch loaded below $\tau$ of its
rating at a snapshot has no influence on the voltage law of that snapshot. The
surviving coefficients are bounded below by
$\tau \cdot 10^4\, \hat{x}^{n,\mathrm{fix}}_\ell F^{n,\mathrm{fix}}_\ell$, i.e.
by $\tau$ times a quantity the impedance scaling (9) leaves invariant under the
iteration, so the range of the constraint matrix no longer degrades from one
iteration to the next. The dropped terms perturb their row by less than $\tau$
of its rated voltage drop. `sensitivity_tolerance=0` keeps every sensitivity.

The residual $\hat{V}^n$ is evaluated on the untruncated branch terms $g^{n,*}$,
so the truncation cannot hide itself from the diagnostics.

### 5.3 Fallback

The linearised LP can be infeasible where the frozen impedances require more
capacity than the linearised voltage law admits. The next iteration is then
taken as a plain fixed-point step, without the linearisation that made the
attempt infeasible, which is always feasible. The same fallback is used if a
solve reports success but returns capacities below their own lower bounds, which
is how a numerically failed solve shows up — such a point must not become a
linearisation point. Both are logged in `n.iteration_log` with `accepted=False`.
Neither applies where no linearisation was in force (`scheme="fixed_point"`, or
the first iteration): an infeasible solve is then raised, and out-of-bound
capacities are only warned about, there being no linearisation to fall back
from.

### 5.4 What the loop returns

Iterations read back only the variable groups the outer loop needs — the
extendable `Line` and `LineX` capacities, the KVL flows `{c}-s` of every
non-empty passive branch component, `LineX-q_sssc` where SSSCs are present,
`Link-p_nom` where DC links are extendable (the report solve pins them, see
below), plus the nominal-capacity columns and objective scalars if
`track_iterations=True` — and neither duals nor derived time series. How the run
is reported then depends on how it ended.

* **On convergence** the run is reported from the iterate it converged on. The
  linearisation is exact at $F^{\mathrm{fix}}$ (property 1 of section 5.1), so
  the model that iterate solved *is* the exact problem at the point it returned
  and a further solve there would only reproduce it. What the iterate lacks is
  coverage, not accuracy, so its own solve is completed in place: every
  remaining primal and every dual is read from the model it was already solved
  on, and the user-facing solution is assigned from it. The model left on
  `n.model` is therefore the iterate's own and still carries the proximal
  penalty, which is subtracted from `n.objective` again — the penalty is a step
  control, not a cost of the system.
* **Without convergence** (`max_iterations` exhausted) the last iterate is a
  point the iteration was still moving away from, and its model has been
  released like every superseded one. The run is closed by a **report solve**:
  the impedances are set from the last capacities $F^{\mathrm{fix}}$, those
  capacities — AC branches and DC links alike — are pinned by
  $F^{\min} = F^{\max} = F^{\mathrm{fix}}$, and the plain formulation (10d) is
  solved once, with neither sensitivities nor penalty. Everything that is not
  transmission capacity — the series compensation, the generation and storage
  capacities, the dispatch — is re-optimised there, which is what makes the
  result the cost *of* that plan rather than of another one. Pinning is what
  keeps it from moving away from the point its own impedances came from. If it
  fails, the status of the last successful loop solve is reported, but no
  solution is left on the network.

The sensitivity attached to the network is cleared on the way out of the loop
in either case, also when it is left through an exception. The capacity bounds
are only ever modified by the report solve, which restores them itself; a
converged run never touches them.

---

## 6. Step control — the proximal term (`proximal`, `proximal_weight`)

The linear model of section 5.1 is exact at its own point and says nothing about
how far from it it may be trusted, so nothing in the inner problem keeps the
capacities near the linearisation point. The proximal term is what bounds the
step. It acts on the extendable `Line` and `LineX` capacities only, and it is
inactive under `scheme="fixed_point"`, which freezes rather than linearises the
voltage law and leaves no linearisation error to control.

It is a *soft* bound: it prices the step instead of forbidding it, so no bound
on the step follows from it alone. What it does is break the degeneracy the
inner problem would otherwise be free in: of two plans of equal system cost it
selects the one closer to the previous iterate, which is the one whose voltage
law the linearisation still describes.

### 6.1 The penalty

The inner objective carries an extra penalty on moving a branch capacity away
from the linearisation point,

$$P^n(F) = \delta \sum_{\ell \in \mathcal{L}^{\mathrm{ext}}} c_\ell\, F^{n,\mathrm{fix}}_\ell\, u_\ell^2,
\qquad u_\ell = \frac{F_\ell - F^{n,\mathrm{fix}}_\ell}{F^{n,\mathrm{fix}}_\ell}.$$

The previous-iteration capacity is used directly as the reference; no separate
scale or capacity floor is introduced. The weight $c_\ell$ is the annualised
capital cost of the branch, so every branch faces the same hurdle relative to
its own cost; a branch whose capital cost is missing or non-positive is charged
the mean of the rest. The penalty charges $\delta c_\ell$ for moving branch
$\ell$ by its own size, so $\delta$ is dimensionless.

**The penalty does not move the converged solution.** Let $J$ be the system
cost, $F^\delta$ minimise $J + P^n$ and $F^\star$ minimise $J$ alone over the
same feasible set. Since $F^\delta$ is optimal for the penalised problem,

$$J(F^\delta) \;\le\; J(F^\delta) + P^n(F^\delta) \;\le\; J(F^\star) + P^n(F^\star),
\qquad\text{hence}\qquad
J(F^\delta) - J(F^\star) \;\le\; P^n(F^\star).$$

The plan the penalty returns costs at most what the penalty charges the
unpenalised plan. That bound is useful because of where it is evaluated: at a
fixed point $F^\star = F^{n,\mathrm{fix}}$, where $P^n$ vanishes. The penalty
biases the *path*, not the limit — it can only shift a solution the iteration
has not converged to. It is also subtracted from the reported system cost, so
the criterion of section 7 sees the true cost throughout.

**Why the quadratic form and not an absolute value.** An $\ell_1$ penalty keeps
the inner problem a linear program, which is why it was the first
implementation, but its subdifferential at the anchor is the whole interval
$\delta c_\ell[-1, 1]$: a branch whose reduced cost is smaller in modulus than
$\delta c_\ell$ does not move **at all**. That dead zone is not a mild bias, it
is a failure mode — once $\delta$ is large enough that the zone covers every
branch, the iterate freezes, and a frozen iterate reports $\mathrm{step}^n = 0$
and $\hat{V}^n = 0$, i.e. it *fabricates* the evidence of convergence at a point
that is not a solution.

The quadratic term has no dead zone. Its gradient $2\delta c_\ell u_\ell$
vanishes at the anchor, so the stationarity conditions of the penalised problem
*at the anchor* are exactly those of the unpenalised one, and every branch moves
by an amount that merely shrinks with $\delta$. The price of a large $\delta$ is
iterations, not accuracy.

**Implementation.** The term introduces a scaled free deviation variable

$$
z_\ell = \sqrt{\frac{\delta c_\ell}{F^{n,\mathrm{fix}}_\ell}}
\left(F_\ell-F^{n,\mathrm{fix}}_\ell\right)
$$

and adds $\sum_\ell z_\ell^2$ to the objective. This makes every diagonal
quadratic coefficient identical and moves the branch-specific scale into a
linear defining equality, which is kinder to the solver than writing the branch
weights into the quadratic objective directly; presolve can eliminate the extra
variables and equalities again.

### 6.2 Raising the weight during the run (`proximal_adaptive`)

With `proximal_adaptive` the weight is not a constant: whenever an accepted
iterate reports that most of the capital its own step and the step before it
moved was moved back again,

$$p^n < \bar{p} \quad\text{and}\quad \delta^n = \delta^{n-1}
\quad\Longrightarrow\quad
\delta \leftarrow \min\!\left(2\delta,\ \delta_{\max}\right),$$

with $\delta_{\max} =$ `proximal_ceiling` and $\bar{p} =$ `PROXIMAL_TURN_BAR`
$= 0.65$. Here $p^n$ is `progress` in `n.iteration_log`, the share of the
capital moved over the last two steps that is net displacement: one where every
branch walked in one direction, zero where every branch came back. It reads the
direction of the movement and its size together, which neither the step nor the
KVL residual does on its own — a turning iteration and a converging one both
produce a falling step, and the residual falls under damping whether or not the
plan has settled. `proximal_weight` is then the *initial* weight.

A **single** reading below the bar is enough. What is required instead is that
the reading be about the plan at all: it spans two steps, and the side condition
$\delta^n = \delta^{n-1}$ drops the readings whose two steps were solved under
different weights, which read low for that reason alone. Two such readings
occur — the second iterate of the run, whose predecessor is solved with no
penalty whatsoever for want of an anchor to write one around, and the iterate
after each doubling. The latter is the cooldown that lets a new penalty take
effect before it is judged again; what it costs is one iterate per doubling on a
run that really is turning, and what it buys is that no doubling is ever ordered
by the arrival of the previous one.

The weight is never released. That is safe for the same reason the whole term
is: a fixed point of the penalised step is a fixed point of the unpenalised
problem, so a weight that ends up too high costs iterations but does not move
the plan; releasing it, by contrast, re-opens the oscillation it was raised to
close.

**The ceiling is not optional.** The rule reads a *direction* rather than a
level, and a direction is meaningless once the steps are numerical noise: the
sign is then random, so half the iterates of a run that has already stopped
would be read as turns and ratchet the weight up. `PROXIMAL_STEP_FLOOR` catches
the clearly dead case — a window moving less than $10^{-9}$ of the plan's own
capital is read as a fixed point rather than as a direction — and the ceiling
bounds whatever gets through. The failure it prevents was measured on the
abandoned residual-rise signal, where every wobble of a residual sitting at the
solver tolerance read as a rise: unbounded, that rule walked the weight from 1
to 512 in thirty iterations on the small meshed test system.

The default ceiling of 4 is what the rule needed on the systems this was
measured on rather than a margin above them: on the 1250 bus case the weight
settled at 3 to 4 and never approached a higher bound.

The rule **costs nothing on a system that never needed it**, which is what the
earlier signals did not manage: on the small meshed system, where a fixed weight
of 1.0 reaches an exact fixed point in 16 iterations, it reproduces that run
iterate for iterate and never raises the weight, `progress` staying between 0.87
and 1.00. Triggering on a rise in the step took 29 iterations there and
triggering on a rise in the KVL residual 30, both by damping a run that was
already converging. It is on by default because the failure it prevents is
silent — a weight below the stability boundary produces a *converged* run whose
plan is still moving, and whose cost is lower than that plan can deliver. Set
`proximal_adaptive=False` where a fixed weight is known to work.

Measured on the 1250 bus case, from the default initial weight of 1.0. These
runs predate the cooldown described above, which delays each doubling by one
iterate; what they establish is the failure the rule prevents rather than the
iteration count it reaches now:

| run | converges at | step | KVL residual |
| --- | --- | --- | --- |
| fixed `delta = 1` | 16 | 0.088 | 1.6e-4 (period-2 orbit) |
| fixed `delta = 2.5`, the best fixed weight there | 22 | 0.0064 | 1.8e-6 |
| adaptive from 1.0, settling at 4 | 18 | 0.0046 | 8.7e-7 |
| adaptive from 1.5, settling at 3 | 17 | 0.0070 | 2.0e-6 |

### 6.3 Choosing the weight, and what is not adapted

Stationarity of the penalised inner problem gives
$u_\ell = -r_\ell / (2 \delta c_\ell)$ for a branch with reduced cost $r_\ell$:
the relative step the penalty admits is of order $1/\delta$, so raising
$\delta$ shortens every step in the way a hard bound on the step would, without
forbidding any particular one.

Without `proximal_adaptive`, $\delta$ is **fixed for the whole run**. Each
system has a stability boundary below which the iteration reflects instead of
contracting — an expanding branch lowers its own impedance and attracts flow,
which the next model answers by shrinking it again — and settles into an orbit
of period two. The boundary is a property of the system, not of the default, so
a run whose `step` stops contracting wants a larger `proximal_weight`;
`proximal_adaptive` above is what detects that state for you.

Raising $\delta$ is safe in a way that bounding the step is not, because a fixed
point of the penalised step is a fixed point of the unpenalised problem: the
converged plan does not depend on the weight the run needed to reach it. A point
the iteration merely *stopped at* is a different matter — a heavier weight
shortens the step, the change of the cost and the KVL residual alike, so all
three convergence signals can be produced by the damping rather than by
stationarity. `step` in `n.iteration_log` is what tells the two apart: a damped
iterate is still moving, a converged one is not.

The penalty also acts as a hurdle rate on capital reallocation, and that is how
it can go wrong: a reallocation between two plans of equal cost gains nothing
and is refused, which is the property that breaks the degeneracy, but a large
enough $\delta$ lets that dead zone decide the plan rather than damp the
iteration.

**Every solved iterate is accepted**, however large the residual it leaves. The
penalty narrows the *next* step only; nothing is ever discarded, and a large
residual by itself never ends the run — the iteration continues until the cost
converges or `max_iterations` runs out. The weight is the only thing that
changes during a run, and only under `proximal_adaptive`.

---

## 7. Convergence criterion

**System cost of an iterate.** Let $z^{n,*}$ be the objective value of the LP,
with the proximal penalty removed again. The cost of the already installed
capacity — over every extendable asset, i.e. the annualised cost coefficient of
(1a) times the existing capacity of generators, storage, AC and DC branches and
SSSCs — is subtracted as determined in the first iteration:

$$obj^n = z^{n,*} - \left(\sum_{g,\,s,\,\ell,\,i} c\,\cdot\,(\text{existing capacity})\right)\Bigg|_{n=1} .$$

Freezing that constant is necessary: for branches with a standard type PyPSA
derives $F_\ell$ from `num_parallel`, so the objective constant that
`n.objective` subtracts drifts with the linearisation capacity and is not
comparable between iterations.

**Criterion.** Over the accepted iterates,

$$\left|obj^{\,n-k} - obj^{\,n-k-1}\right| \Big/ \left|obj^{\,n}\right| \;\le\; \varepsilon
\qquad \text{for } k = 0, \dots, \texttt{cost\_window} - 1 ,$$

with $\varepsilon =$ `cost_threshold` (default $10^{-5}$) and `cost_window`
defaulting to 2. This is the whole criterion; `min_iterations` is the only other
thing that can hold the run open.

**Why not the capacity change.** Algorithm 1 stops on
$\mathrm{step}^n = \lVert F^{n,*} - F^{n,\mathrm{fix}} \rVert_1 / \lVert F^0 \rVert_1$,
unweighted, but that is a step size, not a measure of convergence. It is not invariant under
the exchange of degenerate alternative optima, so it keeps bouncing long after
the solution has settled, and a step can undercut a threshold by accident while
the cost is still drifting. The cost, by contrast, is invariant under the
exchange of degenerate optima and is the quantity the results are reported in.
`step` is still logged, because it is a useful diagnostic of degeneracy and the
only signal that separates a converged iterate from a heavily damped one.
