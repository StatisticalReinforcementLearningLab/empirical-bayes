"""
fit_heartsteps.py
-----------------
Route 1 (fit-then-simulate) preprocessing for the two-variable contextual bandit.

Reads the HeartSteps CSVs, fits the hierarchical reward model described in Section
4 of the write-up, and prints the fitted environment parameters
    mu_a, Sigma_a, sigma^2_a, p_c, p_s
as a paste-ready C++ block for t_sweep_two_variable_hierarchical.cpp, plus a
plain-number summary.

WHAT YOU MUST EDIT: the CONFIG block below. This script cannot know your real
column names or your choice of c_i / s_{i,t}; those are modeling decisions.
Everything below CONFIG is generic and does not need editing.

Estimation (per arm a), matching the write-up:
  - Per participant i, fit the folded 2D model R ~ (1, s) by ridge, giving the
    folded intercept alpha_i = theta_{i,0}+theta_{i,1} c_i, the state slope
    beta_i = theta_{i,2}, and their within-person covariance.
  - EM over the population hyperprior recovers mu_a and the IDENTIFIABLE entries
    of Sigma_a (tau^2_00, tau^2_22, tau_02); this is the same E/M step as the
    hierarchical strategy, run once offline (Section 4.2).
  - The context variance tau^2_11 is NOT identifiable from within-person data
    (c_i is constant within a participant), so it is recovered by the
    between-group method of moments (Section 4.4): split participants by c_i and
    attribute the extra between-person spread in the c=1 group to tau^2_11.
  - Context off-diagonals (tau_01, tau_12) are set to 0 (identifiability constraint).
  - sigma^2_a is the per-arm residual variance with the (N_a - p) correction (BTK 14/15).
"""

import os
import sys
import numpy as np
import pandas as pd

# =====================================================================
# CONFIG  --  EDIT THIS BLOCK ONLY
# =====================================================================
# File paths to the HeartSteps CSVs.

path_files = "/Users/betinakreiman/Desktop/Research/Online-Learning-in-Bandits-with-Predicted-Context/HeartStepV1/HeartStepsV1-main/data_files/"

SUGGESTIONS_CSV = path_files + "suggestions.csv"   # decision-level table (8274 x 86)
USERS_CSV       = path_files + "users.csv"         # participant-level table (37 x 117)
STEPS_CSV       = path_files + "jbsteps.csv"       # minute-level steps (or gfsteps.csv)

# --- identifier columns (join keys) ---
USER_ID_COL     = "user.index"        # participant id, present in every table
DECISION_ID_COL = "decision.index"    # decision-point id in suggestions.csv

# --- ACTION a in {0,1}: whether an (active) suggestion was sent ---
# Set this to the column in suggestions.csv that encodes send/no-send, and the
# value that counts as "sent". If the column is already 0/1, set SENT_VALUE=1.
ACTION_COL      = "send"              # e.g. "send" or "notification.sent"
SENT_VALUE      = True                   # value of ACTION_COL that means "suggestion sent"

# --- REWARD Y: post-decision step count ---
# Two supported modes:
#   mode "column": suggestions.csv already has an aggregated post-decision step
#                  count (the screenshot mentions "aggregated step counts before
#                  and after the suggestions"). Give that column name.
#   mode "window": compute Y by summing STEPS_CSV over a window after each
#                  decision time. Requires timestamp columns; see build_reward().
REWARD_MODE     = "column"
STEP_COUNT_COL  = "jbsteps30"       # used when REWARD_MODE == "column"
# window-mode settings (only used if REWARD_MODE == "window"):
DECISION_TIME_COL = "decision.utime"  # timestamp of the decision in suggestions.csv
STEP_TIME_COL     = "steps.utime"     # timestamp column in STEPS_CSV
STEP_VALUE_COL    = "steps"           # step-count column in STEPS_CSV
WINDOW_MINUTES    = 30                # sum steps in [decision_time, +WINDOW_MINUTES]

# --- CONTEXT c_i (time-invariant, from users.csv), binarized to {0,1} ---
# Pick a stable participant trait with real between-person spread.
CONTEXT_COL       = "consc"   # column in users.csv
# How to binarize: "median" (>= median -> 1) or "threshold" (>= CONTEXT_THRESHOLD -> 1)
CONTEXT_BINARIZE  = "median"
CONTEXT_THRESHOLD = None

# --- STATE s_{i,t} (time-varying, from suggestions.csv), binarized to {0,1} ---
# Pick a decision-level variable with real within-person variation.
STATE_COL         = "state_bin"    # column in suggestions.csv
STATE_BINARIZE    = "asis"              # "median", "threshold", or "asis" (already 0/1)
STATE_THRESHOLD   = None

# --- ridge penalty for the per-person folded fit (match the .cpp lam) ---
LAM = 1e-6
# =====================================================================
# END CONFIG
# =====================================================================

P = 3  # coefficients per participant in the full model (intercept, context, state)


# ---------------------------------------------------------------------
# Data loading and analysis-table construction
# ---------------------------------------------------------------------
def _binarize(series, how, threshold):
    x = pd.to_numeric(series, errors="coerce")
    if how == "asis":
        return (x != 0).astype(int)
    if how == "median":
        med = x.median()
        return (x >= med).astype(int)
    if how == "threshold":
        if threshold is None:
            raise ValueError("threshold binarize requested but threshold is None")
        return (x >= threshold).astype(int)
    raise ValueError(f"unknown binarize mode {how}")


def build_reward(sugg):
    """Return a Series of post-decision step counts Y aligned to `sugg` rows."""
    if REWARD_MODE == "column":
        if STEP_COUNT_COL not in sugg.columns:
            raise KeyError(
                f"REWARD_MODE='column' but '{STEP_COUNT_COL}' not in suggestions.csv. "
                f"Set STEP_COUNT_COL, or switch REWARD_MODE to 'window'."
            )
        return pd.to_numeric(sugg[STEP_COUNT_COL], errors="coerce")

    # window mode: sum STEPS_CSV over [decision_time, decision_time + WINDOW]
    steps = pd.read_csv(STEPS_CSV)
    for col in (USER_ID_COL, STEP_TIME_COL, STEP_VALUE_COL):
        if col not in steps.columns:
            raise KeyError(f"'{col}' not in {STEPS_CSV} (needed for window mode).")
    Y = np.full(len(sugg), np.nan)
    steps_by_user = {uid: g for uid, g in steps.groupby(USER_ID_COL)}
    win = WINDOW_MINUTES * 60.0  # assume unix seconds; adjust if minutes/other
    for r in range(len(sugg)):
        uid = sugg.iloc[r][USER_ID_COL]
        t0 = sugg.iloc[r][DECISION_TIME_COL]
        g = steps_by_user.get(uid)
        if g is None or pd.isna(t0):
            continue
        m = (g[STEP_TIME_COL] >= t0) & (g[STEP_TIME_COL] < t0 + win)
        Y[r] = g.loc[m, STEP_VALUE_COL].sum()
    return pd.Series(Y, index=sugg.index)


def build_analysis_table():
    sugg = pd.read_csv(SUGGESTIONS_CSV)
    sugg["state_bin"] = (sugg["dec.location.category"] == "home").astype(int)
    users = pd.read_csv(USERS_CSV)

    for col, tbl, name in [(USER_ID_COL, sugg, "suggestions"),
                           (ACTION_COL, sugg, "suggestions"),
                           (STATE_COL, sugg, "suggestions"),
                           (USER_ID_COL, users, "users"),
                           (CONTEXT_COL, users, "users")]:
        if col not in tbl.columns:
            raise KeyError(f"Column '{col}' not found in {name}.csv. Edit CONFIG.")

    # action
    A = (sugg[ACTION_COL] == SENT_VALUE).astype(int)

    # reward
    Y = build_reward(sugg)
    R = np.log(Y + 0.5)

    # state (decision-level), binarized
    s = _binarize(sugg[STATE_COL], STATE_BINARIZE, STATE_THRESHOLD)

    # context (participant-level), binarized, then merged onto decisions
    users = users[[USER_ID_COL, CONTEXT_COL]].copy()
    users["c_i"] = _binarize(users[CONTEXT_COL], CONTEXT_BINARIZE, CONTEXT_THRESHOLD)
    cmap = dict(zip(users[USER_ID_COL], users["c_i"]))

    df = pd.DataFrame({
        "user": sugg[USER_ID_COL].values,
        "A": A.values,
        "R": R.values,
        "s": s.values,
    })
    df["c"] = df["user"].map(cmap)

    # drop rows with missing reward / covariates
    df = df.dropna(subset=["R", "A", "s", "c"]).reset_index(drop=True)
    df["A"] = df["A"].astype(int)
    df["s"] = df["s"].astype(int)
    df["c"] = df["c"].astype(int)
    return df


# ---------------------------------------------------------------------
# Per-participant folded fit: R ~ (1, s)  ->  alpha_i, beta_i, within-person cov
# ---------------------------------------------------------------------
def fit_participant_folded(rows):
    """Ridge fit of R on (1, s) for one participant-arm.
    Returns (theta2 [alpha,beta], cov2 2x2, n_obs, rss)."""
    X = np.column_stack([np.ones(len(rows)), rows["s"].values.astype(float)])
    y = rows["R"].values.astype(float)
    XtX = X.T @ X
    G = XtX + LAM * np.eye(2)
    Ginv = np.linalg.inv(G)
    theta = Ginv @ (X.T @ y)
    resid = y - X @ theta
    n = len(rows)
    dof = max(n - 2, 1)
    s2 = float(resid @ resid) / dof
    cov2 = s2 * Ginv          # 2x2 within-person coefficient covariance
    return theta, cov2, n, float(resid @ resid)


# ---------------------------------------------------------------------
# EM over the population hyperprior (folded 2D space), Section 4.2 / 1.3.4
# operating on the folded coefficients (alpha, beta). Returns fixed effects and
# the identifiable covariance pieces in folded space.
# ---------------------------------------------------------------------
def em_folded(thetas, covs, max_iter=500, tol=1e-9):
    """thetas: (m,2) per-participant folded estimates; covs: list of 2x2.
    Returns mu2 (2,), Sigma2 (2,2) population hyperparameters in folded space."""
    m = len(thetas)
    mu = thetas.mean(axis=0)
    Sigma = np.cov(thetas.T) + 1e-6 * np.eye(2)
    for _ in range(max_iter):
        Sinv = np.linalg.inv(Sigma)
        mu_new = np.zeros(2)
        second = np.zeros((2, 2))
        for i in range(m):
            Ci_inv = np.linalg.inv(covs[i] + 1e-12 * np.eye(2))
            V = np.linalg.inv(Sinv + Ci_inv)          # E-step posterior cov
            mvec = V @ (Sinv @ mu + Ci_inv @ thetas[i])  # E-step posterior mean
            mu_new += mvec
            second += V + np.outer(mvec, mvec)
        mu_new /= m
        Sigma_new = second / m - np.outer(mu_new, mu_new)
        # PSD floor
        w, Q = np.linalg.eigh(0.5 * (Sigma_new + Sigma_new.T))
        w = np.maximum(w, 1e-8)
        Sigma_new = Q @ np.diag(w) @ Q.T
        if (np.abs(mu_new - mu).sum()
                + np.abs(np.diag(Sigma_new) - np.diag(Sigma)).sum()) < tol:
            mu, Sigma = mu_new, Sigma_new
            break
        mu, Sigma = mu_new, Sigma_new
    return mu, Sigma


# ---------------------------------------------------------------------
# Full per-arm fit: assemble mu_a (3,), Sigma_a (3,3), sigma^2_a
# ---------------------------------------------------------------------
def fit_arm(df_arm):
    """df_arm: analysis-table rows for one arm. Returns mu3, Sigma3, sigma2."""
    # per-participant folded fits
    users = sorted(df_arm["user"].unique())
    thetas, covs, cvals = [], [], []
    total_rss, total_obs = 0.0, 0
    per_user = {}
    for u in users:
        rows = df_arm[df_arm["user"] == u]
        if len(rows) < 1:
            continue
        th, cov, n, rss = fit_participant_folded(rows)
        thetas.append(th); covs.append(cov)
        cvals.append(int(rows["c"].iloc[0]))          # c_i constant within user
        per_user[u] = (th, cov, n)
        total_rss += rss; total_obs += n
    thetas = np.array(thetas)
    cvals = np.array(cvals)

    # --- fixed effects and identifiable covariance via EM in folded space ---
    mu_fold, Sigma_fold = em_folded(thetas, covs)
    # folded space: index 0 = folded intercept (alpha), 1 = state slope (beta)

    # Split fixed effects into the 3D (intercept, context, state) form.
    # E[alpha | c=0] = mu_0 ; E[alpha | c=1] = mu_0 + mu_1 ; E[beta] = mu_2
    a0 = thetas[cvals == 0, 0]
    a1 = thetas[cvals == 1, 0]
    if len(a0) == 0 or len(a1) == 0:
        raise ValueError(
            "Need participants in BOTH context groups (c=0 and c=1) to identify "
            "the context effect. Check CONTEXT_COL / binarization."
        )
    mu0 = a0.mean()
    mu1 = a1.mean() - mu0
    mu2 = mu_fold[1]
    mu3 = np.array([mu0, mu1, mu2])

    # --- identifiable covariance entries ---
    # tau^2_00 (intercept var): between-person var of alpha in c=0 group,
    # minus average within-person estimation variance of alpha in that group.
    def within_var_alpha(mask):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return 0.0
        return np.mean([covs[i][0, 0] for i in idx])

    def within_var_beta(mask):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return 0.0
        return np.mean([covs[i][1, 1] for i in idx])

    def within_cov_ab(mask):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return 0.0
        return np.mean([covs[i][0, 1] for i in idx])

    tau2_00 = max(np.var(a0, ddof=1) - within_var_alpha(cvals == 0), 1e-8) \
        if len(a0) > 1 else 1e-8
    # tau^2_22 (state slope var): pooled across all participants
    beta_all = thetas[:, 1]
    tau2_22 = max(np.var(beta_all, ddof=1) - within_var_beta(np.ones(len(thetas), bool)),
                  1e-8) if len(beta_all) > 1 else 1e-8
    # tau_02 (intercept-state cov): from c=0 group, where alpha = theta_0
    if len(a0) > 1:
        emp_cov = np.cov(np.vstack([thetas[cvals == 0, 0],
                                    thetas[cvals == 0, 1]]))[0, 1]
        tau_02 = emp_cov - within_cov_ab(cvals == 0)
    else:
        tau_02 = 0.0

    # --- context variance tau^2_11 via BETWEEN-GROUP MOM (Section 4.4) ---
    # Var(alpha | c=1) = Var(theta_0 + theta_1) = tau^2_00 + tau^2_11 (tau_01=0).
    if len(a1) > 1:
        var_alpha_c1 = np.var(a1, ddof=1) - within_var_alpha(cvals == 1)
        tau2_11 = max(var_alpha_c1 - tau2_00, 1e-8)
    else:
        tau2_11 = 1e-8

    # --- assemble Sigma_a (context off-diagonals zeroed) ---
    Sigma3 = np.array([
        [tau2_00, 0.0,     tau_02],
        [0.0,     tau2_11, 0.0   ],
        [tau_02,  0.0,     tau2_22],
    ])
    # final PSD floor on the 3x3
    w, Q = np.linalg.eigh(0.5 * (Sigma3 + Sigma3.T))
    w = np.maximum(w, 1e-8)
    Sigma3 = Q @ np.diag(w) @ Q.T

    # --- residual variance sigma^2_a with (N_a - p) correction (BTK 14/15) ---
    sigma2 = total_rss / max(total_obs - P, 1)

    return mu3, Sigma3, sigma2, len(users)


# ---------------------------------------------------------------------
# Prevalences (Section 4.6), written without the # notation (BTK 17)
# ---------------------------------------------------------------------
def prevalences(df):
    # p_c: between-participant proportion with c_i = 1
    per_user_c = df.groupby("user")["c"].first()
    p_c = float((per_user_c == 1).mean())
    # p_s: fraction of all decision points with s = 1
    p_s = float((df["s"] == 1).mean())
    return p_c, p_s


# ---------------------------------------------------------------------
# Output: paste-ready C++ + summary
# ---------------------------------------------------------------------
def emit_cpp(mu, Sigma, sigma2_arms, p_c, p_s):
    def vec3(v):
        return "{%.6f, %.6f, %.6f}" % (v[0], v[1], v[2])

    def m3(S):
        # build an M3 by field assignment so it compiles regardless of ctor
        lines = []
        for i in range(3):
            for j in range(3):
                lines.append("    S.m[%d][%d] = %.6f;" % (i, j, S[i][j]))
        return "\n".join(lines)

    sigma_pooled = np.sqrt(np.mean(sigma2_arms))  # SD, equal-noise fallback
    print("\n// ===== paste into main() of t_sweep_two_variable_hierarchical.cpp =====")
    print("// Fitted from HeartSteps (Route 1). Replaces the hand-chosen conjecture values.")
    print("std::vector<Vec> mu_a = {")
    print("    %s,   // arm 0" % vec3(mu[0]))
    print("    %s    // arm 1" % vec3(mu[1]))
    print("};")
    print("std::vector<M3> Sigma_a(2);")
    print("{ M3& S = Sigma_a[0];")
    print(m3(Sigma[0]))
    print("}")
    print("{ M3& S = Sigma_a[1];")
    print(m3(Sigma[1]))
    print("}")
    print("double p_context = %.6f;" % p_c)
    print("double p_state   = %.6f;" % p_s)
    print("// Per-arm reward-noise SD (BTK 15). To USE per-arm noise, change the")
    print("// reward line in each algorithm to  mean_r + sigma_r_arm[a_sel]*nd(rng);")
    print("double sigma_r_arm[2] = {%.6f, %.6f};" %
          (np.sqrt(sigma2_arms[0]), np.sqrt(sigma2_arms[1])))
    print("// Equal-noise fallback (matches Section 2 convention, single sigma_r):")
    print("double sigma_r = %.6f;" % sigma_pooled)
    print("// ======================================================================\n")


def main():
    df = build_analysis_table()
    print("Analysis table: %d decision points, %d participants, "
          "%d on arm 0 / %d on arm 1"
          % (len(df), df["user"].nunique(),
             (df["A"] == 0).sum(), (df["A"] == 1).sum()))

    results = {}
    sigma2_arms = [None, None]
    mu_arms = [None, None]
    Sigma_arms = [None, None]
    for a in (0, 1):
        df_a = df[df["A"] == a]
        mu3, Sigma3, sigma2, n_users = fit_arm(df_a)
        mu_arms[a], Sigma_arms[a], sigma2_arms[a] = mu3, Sigma3, sigma2
        print("\n--- arm %d (%d participants) ---" % (a, n_users))
        print("mu_%d      = [% .4f, % .4f, % .4f]" % (a, mu3[0], mu3[1], mu3[2]))
        print("Sigma_%d   =\n%s" % (a, np.array2string(Sigma3, precision=4)))
        print("sigma^2_%d = %.4f   (sigma_%d = %.4f)"
              % (a, sigma2, a, np.sqrt(sigma2)))

    p_c, p_s = prevalences(df)
    print("\np_c = %.4f   p_s = %.4f" % (p_c, p_s))

    # heterogeneity-to-noise ratios (Section 4.5)
    for a in (0, 1):
        sd = np.sqrt(sigma2_arms[a])
        print("arm %d: tau_state/sigma  = %.3f    tau_context/sigma = %.3f"
              % (a, np.sqrt(Sigma_arms[a][2, 2]) / sd,
                 np.sqrt(Sigma_arms[a][1, 1]) / sd))

    emit_cpp(mu_arms, Sigma_arms, sigma2_arms, p_c, p_s)


if __name__ == "__main__":
    main()