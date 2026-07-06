// TWO-VARIABLE CONTEXTUAL BANDIT: fixed context c_i + time-varying state s_{i,t}
// Built from t_sweep_scenario1.cpp and t_sweep_scenario2.cpp.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <numeric>
#include <iomanip>
#include <string>

using Vec  = std::vector<double>;
using IVec = std::vector<int>;
using IMat = std::vector<IVec>;

// ---------------- 2x2 linear algebra helpers ----------------
struct M2 { double a,b,c,d; };

inline M2 inv2(const M2& M, double jitter=1e-12) {
    double det = M.a*M.d - M.b*M.c;
    if (std::abs(det) < jitter) det = (det>=0?jitter:-jitter);
    double inv = 1.0/det;
    return {M.d*inv, -M.b*inv, -M.c*inv, M.a*inv};
}
inline M2 add2(const M2& A, const M2& B){ return {A.a+B.a,A.b+B.b,A.c+B.c,A.d+B.d}; }
inline M2 scale2(const M2& A, double s){ return {A.a*s,A.b*s,A.c*s,A.d*s}; }
inline Vec matvec2(const M2& M, const Vec& x){ return {M.a*x[0]+M.b*x[1], M.c*x[0]+M.d*x[1]}; }
inline double quad2(const Vec& x, const M2& M){
    return x[0]*(M.a*x[0]+M.b*x[1]) + x[1]*(M.c*x[0]+M.d*x[1]);
}
inline M2 outer2(const Vec& x){ return {x[0]*x[0], x[0]*x[1], x[1]*x[0], x[1]*x[1]}; }

inline Vec sample_mvn2(const Vec& mu, const M2& S, std::mt19937_64& rng) {
    double l11 = std::sqrt(std::max(S.a, 0.0));
    double l21 = (l11>1e-14) ? S.c/l11 : 0.0;
    double l22 = std::sqrt(std::max(S.d - l21*l21, 0.0));
    std::normal_distribution<double> nd(0.0,1.0);
    double z1=nd(rng), z2=nd(rng);
    return {mu[0]+l11*z1, mu[1]+l21*z1+l22*z2};
}

// ---------------- 3x3 linear algebra helpers ----------------
struct M3 { double m[3][3]; };

inline M3 zero3(){ M3 A{}; return A; }
inline M3 eye3(double v=1.0){ M3 A{}; for (int i=0;i<3;++i) A.m[i][i]=v; return A; }
inline M3 diag3(double x0, double x1, double x2){ M3 A{}; A.m[0][0]=x0; A.m[1][1]=x1; A.m[2][2]=x2; return A; }
inline M3 add3(const M3& A, const M3& B){ M3 C{}; for(int i=0;i<3;++i) for(int j=0;j<3;++j) C.m[i][j]=A.m[i][j]+B.m[i][j]; return C; }
inline M3 sub3(const M3& A, const M3& B){ M3 C{}; for(int i=0;i<3;++i) for(int j=0;j<3;++j) C.m[i][j]=A.m[i][j]-B.m[i][j]; return C; }
inline M3 scale3(const M3& A, double s){ M3 C{}; for(int i=0;i<3;++i) for(int j=0;j<3;++j) C.m[i][j]=s*A.m[i][j]; return C; }
inline Vec matvec3(const M3& A, const Vec& x){
    return {A.m[0][0]*x[0]+A.m[0][1]*x[1]+A.m[0][2]*x[2],
            A.m[1][0]*x[0]+A.m[1][1]*x[1]+A.m[1][2]*x[2],
            A.m[2][0]*x[0]+A.m[2][1]*x[1]+A.m[2][2]*x[2]};
}
inline double quad3(const Vec& x, const M3& A){
    Vec y = matvec3(A,x);
    return x[0]*y[0]+x[1]*y[1]+x[2]*y[2];
}
inline M3 outer3(const Vec& x){
    M3 A{}; for(int i=0;i<3;++i) for(int j=0;j<3;++j) A.m[i][j]=x[i]*x[j]; return A;
}

inline M3 inv3(M3 A, double jitter=1e-10) {
    // Gauss-Jordan inverse with small diagonal jitter if needed.
    for (int i=0;i<3;++i) A.m[i][i] += jitter;
    M3 B = eye3();
    for (int col=0; col<3; ++col) {
        int pivot = col;
        for (int r=col+1; r<3; ++r)
            if (std::abs(A.m[r][col]) > std::abs(A.m[pivot][col])) pivot = r;
        if (pivot != col) {
            for (int k=0;k<3;++k) {
                std::swap(A.m[col][k], A.m[pivot][k]);
                std::swap(B.m[col][k], B.m[pivot][k]);
            }
        }
        double p = A.m[col][col];
        if (std::abs(p) < jitter) p = (p>=0?jitter:-jitter);
        for (int k=0;k<3;++k) { A.m[col][k] /= p; B.m[col][k] /= p; }
        for (int r=0;r<3;++r) if (r!=col) {
            double f = A.m[r][col];
            for (int k=0;k<3;++k) {
                A.m[r][k] -= f*A.m[col][k];
                B.m[r][k] -= f*B.m[col][k];
            }
        }
    }
    return B;
}

inline M3 project_psd3(const M3& A, double floor=1e-8) {
    // Simple stabilizer: symmetrize and floor only the diagonal. This is enough
    // for the diagonal simulation environments below and keeps code compact.
    M3 B{};
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) B.m[i][j] = 0.5*(A.m[i][j]+A.m[j][i]);
    for (int i=0;i<3;++i) B.m[i][i] = std::max(B.m[i][i], floor);
    return B;
}

inline Vec sample_mvn3(const Vec& mu, const M3& S, std::mt19937_64& rng) {
    // Cholesky with diagonal floors.
    double L[3][3] = {{0}};
    for (int i=0;i<3;++i) {
        for (int j=0;j<=i;++j) {
            double sum = S.m[i][j];
            for (int k=0;k<j;++k) sum -= L[i][k]*L[j][k];
            if (i==j) L[i][j] = std::sqrt(std::max(sum, 1e-12));
            else      L[i][j] = sum / std::max(L[j][j], 1e-12);
        }
    }
    std::normal_distribution<double> nd(0.0,1.0);
    double z[3] = {nd(rng), nd(rng), nd(rng)};
    Vec out(3,0.0);
    for (int i=0;i<3;++i) {
        out[i] = mu[i];
        for (int k=0;k<=i;++k) out[i] += L[i][k]*z[k];
    }
    return out;
}

inline double normal_cdf(double z) {
    return 0.5*(1.0+std::erf(z/std::sqrt(2.0)));
}

// ---------------- Environment: fixed c_i and time-varying s_{i,t} ----------------
struct Env {
    std::vector<std::vector<Vec>> theta; // theta[i][a] = (theta_{i,a,0}, theta_{i,a,1}, theta_{i,a,2})
    IVec context;                         // c_i: fixed per participant
    IMat state;                           // s_{i,t}: time varying
};

Env set_environment(int n, int T, const std::vector<Vec>& mu_a,
                    const std::vector<M3>& Sigma_a,
                    double p_context, double p_state,
                    std::mt19937_64& rng) {
    Env env;
    env.theta.assign(n, std::vector<Vec>(2, Vec(3,0.0)));
    for (int a=0; a<2; ++a)
        for (int i=0; i<n; ++i)
            env.theta[i][a] = sample_mvn3(mu_a[a], Sigma_a[a], rng);

    env.context.assign(n, 0);
    std::bernoulli_distribution bd_c(p_context);
    for (int i=0; i<n; ++i) env.context[i] = bd_c(rng) ? 1 : 0; // one fixed per participant

    env.state.assign(n, IVec(T,0));
    std::bernoulli_distribution bd_s(p_state);

    for (int i=0; i<n; ++i)
        for (int t=0; t<T; ++t)
            env.state[i][t] = bd_s(rng) ? 1 : 0;
    return env;
}

inline double expected_reward(const Env& env, int i, int t, int a) {
    double c = env.context[i];
    double s = env.state[i][t];
    return env.theta[i][a][0] + env.theta[i][a][1]*c + env.theta[i][a][2]*s;
}
inline double best_expected(const Env& env, int i, int t) {
    return std::max(expected_reward(env,i,t,0), expected_reward(env,i,t,1));
}


 // TESTING THE POLICY!!!

inline double treatment_effect_cell(const Env& env, int i, int c, int s) {
    return (env.theta[i][1][0] - env.theta[i][0][0])
         + (env.theta[i][1][1] - env.theta[i][0][1]) * c
         + (env.theta[i][1][2] - env.theta[i][0][2]) * s;
}

inline int optimal_action_cell(const Env& env, int i, int c, int s) {
    double tau = treatment_effect_cell(env, i, c, s);
    return (tau >= 0.0) ? 1 : 0;
}

inline double safe_div(double num, double den) {
    return (den > 0.0) ? num / den : 0.0;
}

inline double sd_from_sums(double sum, double sumsq, int n) {
    if (n <= 1) return 0.0;
    double mean = sum / n;
    double var = sumsq / n - mean * mean;
    return std::sqrt(std::max(0.0, var));
}


void write_optimal_action_header(std::ofstream& out) {
    out << "conjecture,run,T,c,s,"
        << "mean_tau_counterfactual,sd_tau_counterfactual,frac_action1_counterfactual,"
        << "n_actual_context,mean_tau_actual,sd_tau_actual,frac_action1_actual,"
        << "obs_count,obs_freq,frac_action1_observed,"
        << "regret_always1_observed,regret_always0_observed\n";
}

void write_optimal_action_diagnostics(std::ofstream& out,
                                      const std::string& conjecture,
                                      int run,
                                      int T,
                                      const Env& env) {
    int n = (int)env.theta.size();

    for (int c = 0; c <= 1; ++c) {
        for (int s = 0; s <= 1; ++s) {

            // Counterfactual summary:
            // Evaluate every participant's theta under this cell (c,s),
            // regardless of their realized context.
            double cf_sum_tau = 0.0;
            double cf_sumsq_tau = 0.0;
            int cf_action1 = 0;

            for (int i = 0; i < n; ++i) {
                double tau = treatment_effect_cell(env, i, c, s);
                cf_sum_tau += tau;
                cf_sumsq_tau += tau * tau;
                if (tau >= 0.0) cf_action1++;
            }

            double cf_mean_tau = cf_sum_tau / n;
            double cf_sd_tau = sd_from_sums(cf_sum_tau, cf_sumsq_tau, n);
            double cf_frac_action1 = (double)cf_action1 / n;

            // Actual-context summary:
            // Only participants whose actual c_i equals this c.
            double act_sum_tau = 0.0;
            double act_sumsq_tau = 0.0;
            int act_n = 0;
            int act_action1 = 0;

            for (int i = 0; i < n; ++i) {
                if (env.context[i] != c) continue;

                double tau = treatment_effect_cell(env, i, c, s);
                act_sum_tau += tau;
                act_sumsq_tau += tau * tau;
                act_n++;
                if (tau >= 0.0) act_action1++;
            }

            double act_mean_tau = safe_div(act_sum_tau, act_n);
            double act_sd_tau = sd_from_sums(act_sum_tau, act_sumsq_tau, act_n);
            double act_frac_action1 = safe_div(act_action1, act_n);

            // Observed-cell summary:
            // Counts actual (c_i, s_{i,t}) pairs in the generated trajectory.
            int obs_count = 0;
            int obs_action1 = 0;
            double regret_always1 = 0.0;
            double regret_always0 = 0.0;

            for (int i = 0; i < n; ++i) {
                if (env.context[i] != c) continue;

                for (int t = 0; t < T; ++t) {
                    if (env.state[i][t] != s) continue;

                    double tau = treatment_effect_cell(env, i, c, s);

                    obs_count++;
                    if (tau >= 0.0) obs_action1++;

                    // If always choose action 1:
                    // regret is positive only when action 0 is truly better.
                    regret_always1 += std::max(0.0, -tau);

                    // If always choose action 0:
                    // regret is positive only when action 1 is truly better.
                    regret_always0 += std::max(0.0, tau);
                }
            }

            double obs_freq = safe_div(obs_count, n * T);
            double obs_frac_action1 = safe_div(obs_action1, obs_count);
            double obs_regret_always1 = safe_div(regret_always1, obs_count);
            double obs_regret_always0 = safe_div(regret_always0, obs_count);

            out << conjecture << ','
                << run << ','
                << T << ','
                << c << ','
                << s << ','
                << cf_mean_tau << ','
                << cf_sd_tau << ','
                << cf_frac_action1 << ','
                << act_n << ','
                << act_mean_tau << ','
                << act_sd_tau << ','
                << act_frac_action1 << ','
                << obs_count << ','
                << obs_freq << ','
                << obs_frac_action1 << ','
                << obs_regret_always1 << ','
                << obs_regret_always0 << '\n';
        }
    }
}


void write_policy_pattern_header(std::ofstream& out) {
    out << "conjecture,run,T,c,n_context,"
        << "frac_opt_00,frac_opt_01,frac_opt_10,frac_opt_11,"
        << "frac_state_dependent\n";
}

void write_policy_pattern_diagnostics(std::ofstream& out,
                                      const std::string& conjecture,
                                      int run,
                                      int T,
                                      const Env& env) {
    int n = (int)env.theta.size();

    for (int c = 0; c <= 1; ++c) {
        int n_context = 0;
        int pattern_count[4] = {0, 0, 0, 0};

        for (int i = 0; i < n; ++i) {
            if (env.context[i] != c) continue;

            int a_s0 = optimal_action_cell(env, i, c, 0);
            int a_s1 = optimal_action_cell(env, i, c, 1);

            int pattern = 2 * a_s0 + a_s1;
            pattern_count[pattern]++;
            n_context++;
        }

        double frac_00 = safe_div(pattern_count[0], n_context); // action 0 in both states
        double frac_01 = safe_div(pattern_count[1], n_context); // action 0 at s=0, action 1 at s=1
        double frac_10 = safe_div(pattern_count[2], n_context); // action 1 at s=0, action 0 at s=1
        double frac_11 = safe_div(pattern_count[3], n_context); // action 1 in both states

        double frac_state_dependent = frac_01 + frac_10;

        out << conjecture << ','
            << run << ','
            << T << ','
            << c << ','
            << n_context << ','
            << frac_00 << ','
            << frac_01 << ','
            << frac_10 << ','
            << frac_11 << ','
            << frac_state_dependent << '\n';
    }
}





// ---------------- 2D ridge state: unpooled individual learning ----------------
// Individual learner uses x^L_{i,t} = (1, s_{i,t}).
// The intercept estimates theta_{i,a,0} + theta_{i,a,1} c_i.
struct RidgeState2 {
    M2 XtX;
    Vec Xty;
    Vec r_hist;
    std::vector<Vec> x_hist;
    int n_obs;
};

inline void init_state2(RidgeState2& st) {
    st.XtX = {0,0,0,0}; 
    st.Xty = {0.0,0.0}; 
    st.n_obs = 0;
}
inline void update_state2(RidgeState2& st, const Vec& x, double r) {
    st.XtX.a += x[0]*x[0]; st.XtX.b += x[0]*x[1];
    st.XtX.c += x[1]*x[0]; st.XtX.d += x[1]*x[1];
    st.Xty[0] += x[0]*r;   st.Xty[1] += x[1]*r;
    st.x_hist.push_back(x); st.r_hist.push_back(r); st.n_obs++;
}
inline void ridge_estimate2(const RidgeState2& st, double lam, const Vec& mu_prior,
                            Vec& theta_hat, M2& Sigma_hat) {
    M2 G = {st.XtX.a+lam, st.XtX.b, st.XtX.c, st.XtX.d+lam};
    M2 Ginv = inv2(G);
    theta_hat = matvec2(Ginv, st.Xty);
    if (st.n_obs == 0) theta_hat = mu_prior;
    double sigma2 = 1.0;
    if (st.n_obs > 2) {
        double rss = 0.0;
        for (int j=0; j<st.n_obs; ++j) {
            double pred = st.x_hist[j][0]*theta_hat[0] + st.x_hist[j][1]*theta_hat[1];
            double e = st.r_hist[j] - pred;
            rss += e*e;
        }
        sigma2 = std::max(rss/(st.n_obs-2), 1e-8);
    }
    Sigma_hat = scale2(Ginv, sigma2);
}

// ---------------- 3D ridge state: pooled learning ----------------
// Pooled learner uses x^P_{i,t} = (1, c_i, s_{i,t}).
struct RidgeState3 {
    M3 XtX;
    Vec Xty;
    Vec r_hist;
    std::vector<Vec> x_hist;
    int n_obs;
};

inline void init_state3(RidgeState3& st) {
    st.XtX = zero3(); 
    st.Xty = Vec(3,0.0); 
    st.n_obs = 0;
}
inline void update_state3(RidgeState3& st, const Vec& x, double r) {
    for (int p=0;p<3;++p) {
        st.Xty[p] += x[p]*r;
        for (int q=0;q<3;++q) st.XtX.m[p][q] += x[p]*x[q];
    }
    st.x_hist.push_back(x); st.r_hist.push_back(r); st.n_obs++;
}
inline void ridge_estimate3(const RidgeState3& st, double lam, const Vec& mu_prior,
                            Vec& theta_hat, M3& Sigma_hat) {
    M3 G = st.XtX;
    for (int k=0;k<3;++k) G.m[k][k] += lam;
    M3 Ginv = inv3(G);
    theta_hat = matvec3(Ginv, st.Xty);
    if (st.n_obs == 0) theta_hat = mu_prior;
    double sigma2 = 1.0;
    if (st.n_obs > 3) {
        double rss = 0.0;
        for (int j=0; j<st.n_obs; ++j) {
            double pred = st.x_hist[j][0]*theta_hat[0]
                        + st.x_hist[j][1]*theta_hat[1]
                        + st.x_hist[j][2]*theta_hat[2];
            double e = st.r_hist[j] - pred;
            rss += e*e;
        }
        sigma2 = std::max(rss/(st.n_obs-3), 1e-8);
    }
    Sigma_hat = scale3(Ginv, sigma2);
}

// ---------------- EB projection helpers ----------------
// D_i^T theta = (theta0 + c_i theta1, theta2).
// If D is 3x2, D z = (z0, c_i z0, z1).
inline M3 D_Si_inv_Dt(int c, const M2& Si_inv) {
    M3 A{};
    A.m[0][0] = Si_inv.a;
    A.m[0][1] = c * Si_inv.a;
    A.m[0][2] = Si_inv.b;
    A.m[1][0] = c * Si_inv.a;
    A.m[1][1] = c * c * Si_inv.a;
    A.m[1][2] = c * Si_inv.b;
    A.m[2][0] = Si_inv.c;
    A.m[2][1] = c * Si_inv.c;
    A.m[2][2] = Si_inv.d;
    return A;
}
inline Vec D_Si_inv_y(int c, const M2& Si_inv, const Vec& y) {
    Vec z = matvec2(Si_inv, y);
    return {z[0], c*z[0], z[1]};
}

// ---------------- Unpooled Thompson Sampling ----------------
double unpooled_ts(int T, int n, const Env& env, double sigma_r, double lam,
                   const Vec& mu_prior2, std::mt19937_64& rng) {
    std::vector<std::vector<RidgeState2>> state(n, std::vector<RidgeState2>(2));
    for (int i=0;i<n;++i) for (int a=0;a<2;++a) init_state2(state[i][a]);

    double total_regret = 0.0;
    std::uniform_int_distribution<int> ud(0,1);
    std::normal_distribution<double> nd(0.0,1.0);

    for (int t=0;t<T;++t) {
        for (int i=0;i<n;++i) {
            Vec xL = {1.0, (double)env.state[i][t]};
            int a_sel;
            if (state[i][0].n_obs == 0 || state[i][1].n_obs == 0) {
                a_sel = ud(rng);
            } else {
                double samp_reward[2];
                for (int a=0;a<2;++a) {
                    Vec th; M2 S;
                    ridge_estimate2(state[i][a], lam, mu_prior2, th, S);
                    Vec draw = sample_mvn2(th, S, rng);
                    samp_reward[a] = xL[0]*draw[0] + xL[1]*draw[1];
                }
                a_sel = (samp_reward[1] > samp_reward[0]) ? 1 : 0;
            }
            double mean_r = expected_reward(env,i,t,a_sel);
            double r = mean_r + sigma_r*nd(rng);
            update_state2(state[i][a_sel], xL, r);
            total_regret += best_expected(env,i,t) - mean_r;
        }
    }
    return total_regret / n;
}

// ---------------- Pooled Thompson Sampling ----------------
// double pooled_ts(int T, int n, const Env& env, double sigma_r, double lam,
//                  const Vec& mu_prior3, std::mt19937_64& rng) {
//     std::vector<RidgeState3> state(2);
//     for (int a=0;a<2;++a) init_state3(state[a]);

//     double total_regret = 0.0;
//     std::uniform_int_distribution<int> ud(0,1);
//     std::normal_distribution<double> nd(0.0,1.0);

//     for (int t=0;t<T;++t) {
//         Vec theta_hat[2]; M3 Sigma_hat[2]; bool ready[2];
//         for (int a=0;a<2;++a) {
//             ridge_estimate3(state[a], lam, mu_prior3, theta_hat[a], Sigma_hat[a]);
//             ready[a] = state[a].n_obs > 0;
//         }
//         for (int i=0;i<n;++i) {
//             Vec xP = {1.0, (double)env.context[i], (double)env.state[i][t]};
//             int a_sel;
//             if (!ready[0] || !ready[1]) {
//                 a_sel = ud(rng);
//             } else {
//                 double samp_reward[2];
//                 for (int a=0;a<2;++a) {
//                     Vec draw = sample_mvn3(theta_hat[a], Sigma_hat[a], rng);
//                     samp_reward[a] = xP[0]*draw[0] + xP[1]*draw[1] + xP[2]*draw[2];
//                 }
//                 a_sel = (samp_reward[1] > samp_reward[0]) ? 1 : 0;
//             }
//             double mean_r = expected_reward(env,i,t,a_sel);
//             double r = mean_r + sigma_r*nd(rng);
//             update_state3(state[a_sel], xP, r);
//             total_regret += best_expected(env,i,t) - mean_r;
//         }
//     }
//     return total_regret / n;
// }

double pooled_ts(int T, int n, const Env& env, double sigma_r, double lam,
    const Vec& mu_prior2, std::mt19937_64& rng) {
    std::vector<RidgeState2> state(2);
    for (int a = 0; a < 2; ++a) init_state2(state[a]);

    double total_regret = 0.0;
    std::normal_distribution<double> nd(0.0, 1.0);

    for (int t = 0; t < T; ++t) {
        Vec theta_hat[2];
        M2 Sigma_hat[2];

        for (int a = 0; a < 2; ++a) {
            ridge_estimate2(state[a], lam, mu_prior2, theta_hat[a], Sigma_hat[a]);
        }

        for (int i = 0; i < n; ++i) {
            Vec xL = {1.0, (double)env.state[i][t]};

            double samp_reward[2];
            for (int a = 0; a < 2; ++a) {
                Vec draw = sample_mvn2(theta_hat[a], Sigma_hat[a], rng);
                samp_reward[a] = xL[0] * draw[0] + xL[1] * draw[1];
            }

            int a_sel = (samp_reward[1] > samp_reward[0]) ? 1 : 0;

            double mean_r = expected_reward(env, i, t, a_sel);
            double r = mean_r + sigma_r * nd(rng);

            update_state2(state[a_sel], xL, r);
            total_regret += best_expected(env, i, t) - mean_r;
        }
    }

    return total_regret / n;
}

// ---------------- Empirical Bayes ----------------
double empirical_bayes(int T, int n, const Env& env, double sigma_r, double lam,
                       const Vec& mu_prior2, const Vec& mu_prior3,
                       std::mt19937_64& rng, double& out_final_shrinkage) {
    std::vector<std::vector<RidgeState2>> state(n, std::vector<RidgeState2>(2));
    for (int i=0;i<n;++i) for (int a=0;a<2;++a) init_state2(state[i][a]);

    Vec mu_pop[2] = {mu_prior3, mu_prior3};
    M3 Sigma_pop[2] = {diag3(1.0,1.0,1.0), diag3(1.0,1.0,1.0)};

    std::vector<std::vector<Vec>> theta_hat(n, std::vector<Vec>(2, Vec(2,0.0)));
    std::vector<std::vector<M2>>  Sigma_hat(n, std::vector<M2>(2, {1,0,0,1}));
    std::vector<std::vector<bool>> has_data(n, std::vector<bool>(2,false));

    double total_regret = 0.0;
    std::uniform_int_distribution<int> ud(0,1);
    std::uniform_real_distribution<double> ur(0.0,1.0);
    std::normal_distribution<double> nd(0.0,1.0);

    const int MAX_ITER = 10;
    const double TOL = 1e-5;

    for (int t=0;t<T;++t) {
        // Step 1: individual 2D ridge estimates.
        for (int i=0;i<n;++i) for (int a=0;a<2;++a) {
            ridge_estimate2(state[i][a], lam, mu_prior2, theta_hat[i][a], Sigma_hat[i][a]);
            has_data[i][a] = state[i][a].n_obs > 0;
        }

        // Step 2: EB hyperparameter update in true 3D theta space.
        for (int a=0;a<2;++a) {
            int nn = 0;
            for (int i=0;i<n;++i) if (has_data[i][a]) nn++;
            if (nn == 0) continue;

            Vec mu_cur = mu_pop[a];
            M3 Sigma_cur = Sigma_pop[a];

            for (int iter=0; iter<MAX_ITER; ++iter) {
                M3 Sp_inv = inv3(Sigma_cur);
                Vec mu_new = {0.0,0.0,0.0};
                M3 second = zero3();

                for (int i=0;i<n;++i) {
                    if (!has_data[i][a]) continue;
                    int c = env.context[i];
                    M2 Si_inv = inv2(Sigma_hat[i][a]);
                    M3 precision = add3(Sp_inv, D_Si_inv_Dt(c, Si_inv));
                    M3 post_cov = inv3(precision);

                    Vec rhs = matvec3(Sp_inv, mu_cur);
                    Vec obs_rhs = D_Si_inv_y(c, Si_inv, theta_hat[i][a]);
                    for (int k=0;k<3;++k) rhs[k] += obs_rhs[k];
                    Vec post_mean = matvec3(post_cov, rhs);

                    for (int k=0;k<3;++k) mu_new[k] += post_mean[k];
                    second = add3(second, add3(post_cov, outer3(post_mean)));
                }
                for (int k=0;k<3;++k) mu_new[k] /= nn;
                M3 Sigma_new = sub3(scale3(second, 1.0/nn), outer3(mu_new));
                Sigma_new = project_psd3(Sigma_new, 1e-8);

                double delta = 0.0;
                for (int k=0;k<3;++k) delta += std::abs(mu_new[k]-mu_cur[k]);
                for (int k=0;k<3;++k) delta += std::abs(Sigma_new.m[k][k]-Sigma_cur.m[k][k]);
                mu_cur = mu_new;
                Sigma_cur = Sigma_new;
                if (delta < TOL) break;
            }
            mu_pop[a] = mu_cur;
            Sigma_pop[a] = Sigma_cur;
        }

        M3 Sp_inv[2] = {inv3(Sigma_pop[0]), inv3(Sigma_pop[1])};
        double step_shrinkage_sum = 0.0;
        int step_shrinkage_count = 0;

        // Step 3: EB posterior action selection.
        for (int i=0;i<n;++i) {
            Vec xL = {1.0, (double)env.state[i][t]};
            Vec xTrue = {1.0, (double)env.context[i], (double)env.state[i][t]};
            int a_sel;
            if (!has_data[i][0] || !has_data[i][1]) {
                a_sel = ud(rng);
            } else {
                double reward_mean[2], reward_var[2];
                for (int a=0;a<2;++a) {
                    int c = env.context[i];
                    M2 Si_inv = inv2(Sigma_hat[i][a]);

                    // shrinkage ratio! 
                    if (t == T-1) {
                        double tr_pop = Sp_inv[a].m[0][0] + Sp_inv[a].m[1][1] + Sp_inv[a].m[2][2];
                        double tr_ind = Si_inv.a + Si_inv.d;
                        step_shrinkage_sum += tr_ind / std::max(tr_pop + tr_ind, 1e-20);
                        step_shrinkage_count++;
                    }

                    M3 precision = add3(Sp_inv[a], D_Si_inv_Dt(c, Si_inv));
                    M3 post_cov = inv3(precision);
                    Vec rhs = matvec3(Sp_inv[a], mu_pop[a]);
                    Vec obs_rhs = D_Si_inv_y(c, Si_inv, theta_hat[i][a]);
                    for (int k=0;k<3;++k) rhs[k] += obs_rhs[k];
                    Vec post_mean = matvec3(post_cov, rhs);

                    reward_mean[a] = xTrue[0]*post_mean[0] + xTrue[1]*post_mean[1] + xTrue[2]*post_mean[2];
                    reward_var[a]  = quad3(xTrue, post_cov);
                }
                double diff = reward_mean[1] - reward_mean[0];
                double sd = std::sqrt(std::max(reward_var[0]+reward_var[1], 1e-20));
                a_sel = (ur(rng) < normal_cdf(diff/sd)) ? 1 : 0;
            }
            double mean_r = expected_reward(env,i,t,a_sel);
            double r = mean_r + sigma_r*nd(rng);
            update_state2(state[i][a_sel], xL, r);
            total_regret += best_expected(env,i,t) - mean_r;
        }

        if (t == T-1 && step_shrinkage_count > 0)
            out_final_shrinkage = step_shrinkage_sum / step_shrinkage_count;
    }
    return total_regret / n;
}

// ---------------- Main ----------------
int main() {
    int n = 100;
    int runs = 20;
    double sigma_r = 0.5;
    double lam = 1e-6;
    double p_context = 0.5;
    double p_state = 0.5;

    // Conjecture 1 baseline: large main/context/state treatment effects.
    std::vector<Vec> mu_a = {
        {0.0, 0.0, 0.0},
        {0.40, 0.20, 0.20}
    };
    std::vector<M3> Sigma_a = {
        diag3(0.50, 0.50, 0.50),
        diag3(0.50, 0.50, 0.50)
    };

    Vec mu_prior2 = {0.0, 0.0};       // learning prior for x^L=(1,s)
    Vec mu_prior3 = {0.0, 0.0, 0.0};  // population/pooled prior for theta=(intercept,c,s)

    IVec T_values;
    for (int Tv=1; Tv<=201; Tv+=10) T_values.push_back(Tv);
    int nT = T_values.size();

    std::ofstream out("conjectures_v2/TESTING_POLICY.csv");
    out << "T,mean_unpooled,se_unpooled,mean_pooled,se_pooled,"
        << "mean_eb,se_eb,final_shrinkage,winner\n";

    std::cout << "Sweeping " << nT << " T values (two-variable environment)...\n";


    // POLICY FILES 

    std::string conjecture_name = "conj1";

    std::ofstream oracle_out("testing_code/conj1_oracle_cells.csv");
    write_optimal_action_header(oracle_out);

    std::ofstream pattern_out("testing_code/conj1_oracle_patterns.csv");
    write_policy_pattern_header(pattern_out);


    for (int ti=0; ti<nT; ++ti) {
        int T = T_values[ti];
        double sum_u=0, sum_p=0, sum_e=0;
        double ssq_u=0, ssq_p=0, ssq_e=0;
        double sum_shrinkage=0.0;

        for (int r=0; r<runs; ++r) {
            std::mt19937_64 rng_env(r + 1000 + ti*10000);
            Env env = set_environment(n, T, mu_a, Sigma_a, p_context, p_state, rng_env);

            write_optimal_action_diagnostics(oracle_out, conjecture_name, r, T, env);
            write_policy_pattern_diagnostics(pattern_out, conjecture_name, r, T, env);

            std::mt19937_64 rng_u(r + 2000 + ti*10000);
            double reg_u = unpooled_ts(T, n, env, sigma_r, lam, mu_prior2, rng_u);

            std::mt19937_64 rng_p(r + 3000 + ti*10000);
            // double reg_p = pooled_ts(T, n, env, sigma_r, lam, mu_prior3, rng_p);
            double reg_p = pooled_ts(T, n, env, sigma_r, lam, mu_prior2, rng_p);

            std::mt19937_64 rng_e(r + 4000 + ti*10000);
            double run_shrinkage = 0.0;
            double reg_e = empirical_bayes(T, n, env, sigma_r, lam,
                                           mu_prior2, mu_prior3, rng_e, run_shrinkage);

            sum_u += reg_u; ssq_u += reg_u*reg_u;
            sum_p += reg_p; ssq_p += reg_p*reg_p;
            sum_e += reg_e; ssq_e += reg_e*reg_e;
            sum_shrinkage += run_shrinkage;
        }

        auto mean_se = [&](double sum, double ssq) -> std::pair<double,double> {
            double mu = sum/runs;
            double var = (ssq - runs*mu*mu)/(runs-1);
            return {mu, std::sqrt(std::max(0.0,var))/std::sqrt(runs)};
        };

        auto [mu_u, se_u] = mean_se(sum_u, ssq_u);
        auto [mu_p, se_p] = mean_se(sum_p, ssq_p);
        auto [mu_e, se_e] = mean_se(sum_e, ssq_e);
        double avg_shrinkage = sum_shrinkage / runs;

        std::string winner;
        double m = std::min({mu_u, mu_p, mu_e});
        if      (mu_u == m) winner="Unpooled";
        else if (mu_p == m) winner="Pooled";
        else                winner="EB";

        out << std::fixed << std::setprecision(4)
            << T << ',' << mu_u << ',' << se_u << ','
            << mu_p << ',' << se_p << ','
            << mu_e << ',' << se_e << ','
            << avg_shrinkage << ',' << winner << '\n';

        std::cout << "T=" << std::fixed << std::setprecision(2) << T
                  << "  U=" << std::setprecision(3) << mu_u
                  << "  P=" << mu_p
                  << "  EB=" << mu_e
                  << "  shrinkage=" << avg_shrinkage
                  << "  winner=" << winner << "\n";
    }

    out.close();


    oracle_out.close();
    pattern_out.close();


    std::cout << "\nDone! Results written to conjectures_v2/TESTING_POLICY.csv\n";
    return 0;
}
