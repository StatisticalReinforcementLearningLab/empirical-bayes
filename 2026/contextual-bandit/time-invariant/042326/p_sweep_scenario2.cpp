#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <numeric>
#include <iomanip>

using Vec  = std::vector<double>;
using IVec = std::vector<int>;

// ---------------- 2x2 linear algebra helpers (identical to scenario 1) ----------------
struct M2 { double a,b,c,d; };

inline M2 inv2(const M2& M, double jitter=1e-12) {
    double det = M.a*M.d - M.b*M.c;
    if (std::abs(det) < jitter) det = (det>=0?jitter:-jitter);
    double inv = 1.0/det;
    return {M.d*inv, -M.b*inv, -M.c*inv, M.a*inv};
}
inline M2 add2(const M2& A, const M2& B){ return {A.a+B.a,A.b+B.b,A.c+B.c,A.d+B.d}; }
inline M2 scale2(const M2& A, double s) { return {A.a*s,A.b*s,A.c*s,A.d*s}; }
inline Vec matvec2(const M2& M, const Vec& x){ return {M.a*x[0]+M.b*x[1], M.c*x[0]+M.d*x[1]}; }
inline double quad2(const Vec& x, const M2& M){
    return x[0]*(M.a*x[0]+M.b*x[1]) + x[1]*(M.c*x[0]+M.d*x[1]);
}
inline M2 outer2(const Vec& x){ return {x[0]*x[0], x[0]*x[1], x[1]*x[0], x[1]*x[1]}; }

inline Vec sample_mvn2(const Vec& mu, const M2& S, std::mt19937_64& rng) {
    double a  = std::sqrt(std::max(S.a, 0.0));
    double l21 = (a>1e-14) ? S.c/a : 0.0;
    double l22 = std::sqrt(std::max(S.d - l21*l21, 0.0));
    std::normal_distribution<double> nd(0.0,1.0);
    double z1=nd(rng), z2=nd(rng);
    return { mu[0]+a*z1, mu[1]+l21*z1+l22*z2 };
}

inline double normal_cdf(double z) {
    return 0.5*(1.0+std::erf(z/std::sqrt(2.0)));
}

// ---------------- Environment (time-invariant context) ----------------
// KEY DIFFERENCE from scenario 1: context c_i is drawn once per participant
// and held fixed for all T time steps (e.g. a demographic variable like sex).
struct Env {
    std::vector<std::vector<Vec>> theta; // theta[i][a] = (theta_{i,a,0}, theta_{i,a,1})
    IVec context;                         // c_i: one value per participant, fixed across time
};

Env set_environment(int n, const std::vector<Vec>& mu_a,
                    const std::vector<M2>& Sigma_a, double p_context,
                    std::mt19937_64& rng) {
    Env env;
    env.theta.assign(n, std::vector<Vec>(2, Vec(2,0.0)));
    for (int a=0; a<2; ++a)
        for (int i=0; i<n; ++i)
            env.theta[i][a] = sample_mvn2(mu_a[a], Sigma_a[a], rng);
    // draw one fixed context per participant
    env.context.assign(n, 0);
    std::bernoulli_distribution bd(p_context);
    for (int i=0; i<n; ++i) env.context[i] = bd(rng) ? 1 : 0;
    return env;
}

// No time index needed: c_i is fixed for participant i
inline double expected_reward(const Env& env, int i, int a) {
    double c = env.context[i];
    return env.theta[i][a][0] + env.theta[i][a][1]*c;
}
inline double best_expected(const Env& env, int i) {
    return std::max(expected_reward(env,i,0), expected_reward(env,i,1));
}

// ---------------- Scalar ridge state (individual 1D learning) ----------------
// KEY DIFFERENCE from scenario 1: because c_i is fixed, each participant only
// ever observes ONE context value.  The context effect cannot be separated from
// the intercept, so the learning algorithm fits a SCALAR model
//   R_{i,t}(a) = theta^L_{i,a} + noise,   theta^L_{i,a} = x_i^T theta_{i,a}
// and outputs a scalar estimate theta_hat^L and scalar uncertainty Sigma^L.
struct ScalarState {
    double sum_r;   // sum of rewards observed for this (participant, arm)
    double sum_r2;  // sum of squared rewards (for residual variance)
    int    n_obs;
};

inline void init_scalar(ScalarState& s){ s.sum_r=0; s.sum_r2=0; s.n_obs=0; }
inline void update_scalar(ScalarState& s, double r){
    s.sum_r+=r; s.sum_r2+=r*r; s.n_obs++;
}

// theta_hat^L = (lam + n)^{-1} * sum_r
// Sigma^L     = sigma2_hat * (lam + n)^{-1},   sigma2_hat via (n-1) denominator
inline void scalar_estimate(const ScalarState& s, double lam,
                             double& th, double& sig) {
    double denom = lam + s.n_obs;
    th = (s.n_obs > 0) ? s.sum_r / denom : 0.0;
    double sigma2 = 1.0;
    if (s.n_obs > 1) {
        double rss = s.sum_r2 - 2.0*th*s.sum_r + (double)s.n_obs*th*th;
        sigma2 = std::max(rss / (s.n_obs - 1), 1e-8);
    }
    sig = sigma2 / denom;
}

// ---------------- 2D ridge state (pooled learning only) ----------------
// Pooled TS can still use a 2D model: it observes x_i = (1, c_i) from many
// participants with different contexts, so it CAN learn the context effect.
struct RidgeState {
    M2   XtX;
    Vec  Xty;
    Vec  r_hist;
    std::vector<Vec> x_hist;
    int  n_obs;
};

inline void init_state(RidgeState& s){
    s.XtX={0,0,0,0}; 
    s.Xty={0.0,0.0}; 
    s.n_obs=0;
}
inline void update_state(RidgeState& s, const Vec& x, double r){
    s.XtX.a+=x[0]*x[0]; s.XtX.b+=x[0]*x[1];
    s.XtX.c+=x[1]*x[0]; s.XtX.d+=x[1]*x[1];
    s.Xty[0]+=x[0]*r;   s.Xty[1]+=x[1]*r;
    s.x_hist.push_back(x); 
    s.r_hist.push_back(r); 
    s.n_obs++;
}

inline void ridge_estimate(const RidgeState& s, double lam, const Vec& mu_prior,
                            Vec& theta_hat, M2& Sigma_hat) {
    M2 G = {s.XtX.a+lam, s.XtX.b, s.XtX.c, s.XtX.d+lam};
    M2 Ginv = inv2(G);
    theta_hat = matvec2(Ginv, s.Xty);
    if (s.n_obs==0) theta_hat = mu_prior;
    double sigma2 = 1.0;
    if (s.n_obs > 2) {
        double rss = 0.0;
        for (int j=0; j<s.n_obs; ++j) {
            double pred = s.x_hist[j][0]*theta_hat[0]+s.x_hist[j][1]*theta_hat[1];
            double e = s.r_hist[j]-pred; rss += e*e;
        }
        sigma2 = std::max(rss/(s.n_obs-2), 1e-8);
    }
    Sigma_hat = scale2(Ginv, sigma2);
}

// ---------------- Unpooled Contextual TS (time-invariant) ----------------
// Uses a scalar estimate per (participant, arm).  Action selection compares
// scalar samples directly—the scalar already encodes the participant's context.
double unpooled_ts(int T, int n, const Env& env, double sigma_r, double lam,
                   std::mt19937_64& rng) {
    std::vector<std::vector<ScalarState>> state(n, std::vector<ScalarState>(2));
    for (int i=0; i<n; ++i) for (int a=0; a<2; ++a) init_scalar(state[i][a]);

    double total_regret = 0.0;
    std::uniform_int_distribution<int> ud(0,1);
    std::normal_distribution<double>   nd(0.0,1.0);

    for (int t=0; t<T; ++t) {
        for (int i=0; i<n; ++i) {
            int a_sel;
            if (state[i][0].n_obs==0 || state[i][1].n_obs==0) {
                // forced exploration until each arm tried at least once
                a_sel = ud(rng);
            } else {
                // sample scalar theta^L for each arm; pick the larger
                double th[2], sig[2], samp[2];
                for (int a=0; a<2; ++a) {
                    scalar_estimate(state[i][a], lam, th[a], sig[a]);
                    std::normal_distribution<double> d(th[a], std::sqrt(sig[a]));
                    samp[a] = d(rng);
                }
                a_sel = (samp[1] > samp[0]) ? 1 : 0;
            }
            double mean_r = expected_reward(env, i, a_sel);
            double r      = mean_r + sigma_r*nd(rng);
            update_scalar(state[i][a_sel], r);
            total_regret += best_expected(env, i) - mean_r;
        }
    }
    return total_regret / n;
}

// ---------------- Pooled Contextual TS (time-invariant) ----------------
// Pools across all participants using x_i = (1, c_i) as a 2D feature vector.
// The shared posterior theta_a ~ N(theta_hat_a, Sigma_a) is 2-dimensional;
// action selection uses x_i^T theta_draw to incorporate each participant's
// fixed context when choosing an arm.
double pooled_ts(int T, int n, const Env& env, double sigma_r, double lam,
                 const Vec& mu0_prior, std::mt19937_64& rng) {
    std::vector<RidgeState> state(2);
    for (int a=0; a<2; ++a) init_state(state[a]);

    double total_regret = 0.0;
    std::uniform_int_distribution<int> ud(0,1);
    std::normal_distribution<double>   nd(0.0,1.0);

    for (int t=0; t<T; ++t) {
        // compute shared posterior once per time step
        Vec theta_hat[2]; M2 Sigma_hat[2]; bool ready[2];
        for (int a=0; a<2; ++a) {
            ridge_estimate(state[a], lam, mu0_prior, theta_hat[a], Sigma_hat[a]);
            ready[a] = state[a].n_obs > 0;
        }
        for (int i=0; i<n; ++i) {
            // x_i = (1, c_i): fixed for this participant
            Vec x = {1.0, (double)env.context[i]};
            int a_sel;
            if (!ready[0] || !ready[1]) {
                a_sel = ud(rng);
            } else {
                double samp[2];
                for (int a=0; a<2; ++a) {
                    Vec th_draw = sample_mvn2(theta_hat[a], Sigma_hat[a], rng);
                    samp[a] = x[0]*th_draw[0] + x[1]*th_draw[1];
                }
                a_sel = (samp[1] > samp[0]) ? 1 : 0;
            }
            double mean_r = expected_reward(env, i, a_sel);
            double r      = mean_r + sigma_r*nd(rng);
            update_state(state[a_sel], x, r);
            total_regret += best_expected(env, i) - mean_r;
        }
    }
    return total_regret / n;
}

// ---------------- Empirical Bayes (time-invariant) ----------------
// Individual learning: scalar estimates theta_hat^L_{i,a,t}, Sigma^L_{i,a,t}.
// Population model: theta_{i,a} ~ N(mu_a, Sigma_a) with Sigma_a DIAGONAL
//   because only two variance equations are identifiable from the two context
//   groups (c_i=0 and c_i=1), so the off-diagonal entry cannot be recovered.
//
// Hyperparameter estimation via MOM (per arm a):
//   tau2_{a,00}: from c_i=0 group  ->  Var(theta_hat^L | c=0) = tau2_00 + mean(Sigma^L)
//   tau2_{a,11}: from c_i=1 group  ->  Var(theta_hat^L | c=1) = tau2_00 + tau2_11 + mean(Sigma^L)
//   mu_hat_a:    closed-form weighted MLE after Sigma_a estimated
//
// Posterior:  theta_{i,a} | theta_hat^L_{i,a,t}
//   post_cov^{-1} = Sigma_a^{-1} + (1/Sigma^L_{i,a,t}) * x_i x_i^T
//   post_mean     = post_cov * (Sigma_a^{-1} mu_a + x_i * theta_hat^L / Sigma^L)
double empirical_bayes(int T, int n, const Env& env, double sigma_r, double lam,
                       const Vec& mu0_prior, std::mt19937_64& rng) {
    std::vector<std::vector<ScalarState>> state(n, std::vector<ScalarState>(2));
    for (int i=0; i<n; ++i) for (int a=0; a<2; ++a) init_scalar(state[i][a]);

    // scalar individual estimates and uncertainties
    std::vector<std::vector<double>> theta_hat(n, std::vector<double>(2, 0.0));
    std::vector<std::vector<double>> sigma_hat(n, std::vector<double>(2, 1.0));
    std::vector<std::vector<bool>>   has_data (n, std::vector<bool>(2, false));

    // population hyperparameters; Sigma_pop kept diagonal
    Vec mu_pop[2]    = {mu0_prior, mu0_prior};
    M2  Sigma_pop[2] = {{1,0,0,1},{1,0,0,1}};

    double total_regret = 0.0;
    std::uniform_int_distribution<int>     ud(0,1);
    std::uniform_real_distribution<double> ur(0.0,1.0);
    std::normal_distribution<double>       nd(0.0,1.0);

    for (int t=0; t<T; ++t) {

        // --- Step 1: update individual scalar estimates ---
        for (int i=0; i<n; ++i) for (int a=0; a<2; ++a) {
            scalar_estimate(state[i][a], lam, theta_hat[i][a], sigma_hat[i][a]);
            has_data[i][a] = state[i][a].n_obs > 0;
        }

        // --- Step 2: MOM hyperparameter estimation per arm ---
        for (int a=0; a<2; ++a) {
            // split participants into c_i=0 and c_i=1 groups (data only)
            std::vector<int> grp0, grp1;
            for (int i=0; i<n; ++i) {
                if (!has_data[i][a]) continue;
                (env.context[i]==0 ? grp0 : grp1).push_back(i);
            }
            int n0=grp0.size(), n1=grp1.size();

            // MOM for tau2_{a,00} using c_i=0 group:
            //   Var(theta_hat^L | c=0) = tau2_00 + mean(Sigma^L),  x_i=(1,0)
            double tau2_00 = 0.0;
            if (n0 > 1) {
                double m0=0; for (int i:grp0) m0+=theta_hat[i][a]; m0/=n0;
                double v0=0; for (int i:grp0){double d=theta_hat[i][a]-m0; v0+=d*d;} v0/=(n0-1);
                double sv0=0; for (int i:grp0) sv0+=sigma_hat[i][a]; sv0/=n0;
                tau2_00 = std::max(v0 - sv0, 0.0);
            }

            // MOM for tau2_{a,11} using c_i=1 group:
            //   Var(theta_hat^L | c=1) = tau2_00 + tau2_11 + mean(Sigma^L),  x_i=(1,1)
            double tau2_11 = 0.0;
            if (n1 > 1) {
                double m1=0; for (int i:grp1) m1+=theta_hat[i][a]; m1/=n1;
                double v1=0; for (int i:grp1){double d=theta_hat[i][a]-m1; v1+=d*d;} v1/=(n1-1);
                double sv1=0; for (int i:grp1) sv1+=sigma_hat[i][a]; sv1/=n1;
                tau2_11 = std::max(v1 - sv1 - tau2_00, 0.0);
            }

            // diagonal Sigma_a; small floor ensures invertibility
            Sigma_pop[a] = { std::max(tau2_00, 1e-8), 0.0,
                             0.0, std::max(tau2_11, 1e-8) };

            // MLE for mu_hat_a (closed form after Sigma_a fixed):
            //   mu_hat = (sum_i x_i x_i^T / v_i)^{-1} (sum_i x_i theta_hat^L_i / v_i)
            //   where v_i = x_i^T Sigma_a x_i + Sigma^L_{i,a}
            M2  A = {0,0,0,0};
            Vec b = {0.0, 0.0};
            for (int i=0; i<n; ++i) {
                if (!has_data[i][a]) continue;
                Vec xi = {1.0, (double)env.context[i]};
                double v_i = quad2(xi, Sigma_pop[a]) + sigma_hat[i][a];
                if (v_i < 1e-20) continue;
                A = add2(A, scale2(outer2(xi), 1.0/v_i));
                b[0] += xi[0]*theta_hat[i][a]/v_i;
                b[1] += xi[1]*theta_hat[i][a]/v_i;
            }
            mu_pop[a] = matvec2(inv2(A), b);
        }

        // --- Step 3: EB posterior and action selection ---
        for (int i=0; i<n; ++i) {
            int a_sel;
            if (!has_data[i][0] || !has_data[i][1]) {
                a_sel = ud(rng);
            } else {
                Vec xi = {1.0, (double)env.context[i]};
                double reward_mean[2], reward_var[2];
                for (int a=0; a<2; ++a) {
                    // posterior covariance:
                    //   post_cov^{-1} = Sigma_a^{-1} + (1/Sigma^L) * x_i x_i^T
                    M2 Sp_inv       = inv2(Sigma_pop[a]);
                    M2 post_cov_inv = add2(Sp_inv,
                                          scale2(outer2(xi), 1.0/sigma_hat[i][a]));
                    M2 post_cov     = inv2(post_cov_inv);
                    // posterior mean:
                    //   post_mean = post_cov * (Sigma_a^{-1} mu_a + x_i theta_hat^L / Sigma^L)
                    Vec rhs = matvec2(Sp_inv, mu_pop[a]);
                    rhs[0] += xi[0]*theta_hat[i][a]/sigma_hat[i][a];
                    rhs[1] += xi[1]*theta_hat[i][a]/sigma_hat[i][a];
                    Vec post_mean = matvec2(post_cov, rhs);
                    // expected reward and variance for arm a given participant's context
                    reward_mean[a] = xi[0]*post_mean[0] + xi[1]*post_mean[1];
                    reward_var[a]  = quad2(xi, post_cov);
                }
                double diff = reward_mean[1]-reward_mean[0];
                double sd   = std::sqrt(std::max(reward_var[0]+reward_var[1], 1e-20));
                a_sel = (ur(rng) < normal_cdf(diff/sd)) ? 1 : 0;
            }
            double mean_r = expected_reward(env, i, a_sel);
            double r      = mean_r + sigma_r*nd(rng);
            update_scalar(state[i][a_sel], r);
            total_regret += best_expected(env, i) - mean_r;
        }
    }
    return total_regret / n;
}

// ---------------- Main ----------------
int main() {
    int    n      = 200;
    int    T      = 30;
    int    runs   = 200;
    double sigma_r = 1.0;
    double lam     = 1e-6;

    // default environment: 
    std::vector<Vec> mu_a = {
        {0.0,  1.0},   
        {0.5, -1.0}    
    };
    std::vector<M2> Sigma_a = {
        {0.1, 0.0, 0.0, 0.1},
        {0.1, 0.0, 0.0, 0.1}
    };
    Vec mu0_prior = {0.0, 0.0};

    Vec p_values;
    for (double p=0.10; p<=0.50+1e-9; p+=0.05) p_values.push_back(p);
    int np = p_values.size();

    std::ofstream out("graphs_ti/highn.csv");
    out << "p_context,mean_unpooled,se_unpooled,mean_pooled,se_pooled,"
           "mean_eb,se_eb,winner\n";

    std::cout << "Sweeping " << np << " p values (time-invariant context)...\n";

    for (int pi=0; pi<np; ++pi) {
        double p = p_values[pi];
        double sum_u=0, sum_p=0, sum_e=0;
        double ssq_u=0, ssq_p=0, ssq_e=0;

        for (int r=0; r<runs; ++r) {
            std::mt19937_64 rng_env(r + 1000 + pi*10000);
            Env env = set_environment(n, mu_a, Sigma_a, p, rng_env);

            std::mt19937_64 rng_u(r + 2000 + pi*10000);
            double reg_u = unpooled_ts(T, n, env, sigma_r, lam, rng_u);

            std::mt19937_64 rng_p(r + 3000 + pi*10000);
            double reg_p = pooled_ts(T, n, env, sigma_r, lam, mu0_prior, rng_p);

            std::mt19937_64 rng_e(r + 4000 + pi*10000);
            double reg_e = empirical_bayes(T, n, env, sigma_r, lam, mu0_prior, rng_e);

            sum_u+=reg_u; ssq_u+=reg_u*reg_u;
            sum_p+=reg_p; ssq_p+=reg_p*reg_p;
            sum_e+=reg_e; ssq_e+=reg_e*reg_e;
        }

        auto mean_se = [&](double sum, double ssq) -> std::pair<double,double> {
            double mu  = sum/runs;
            double var = (ssq - runs*mu*mu)/(runs-1);
            return {mu, std::sqrt(std::max(0.0,var))/std::sqrt(runs)};
        };

        auto [mu_u, se_u] = mean_se(sum_u, ssq_u);
        auto [mu_p, se_p] = mean_se(sum_p, ssq_p);
        auto [mu_e, se_e] = mean_se(sum_e, ssq_e);

        std::string winner;
        double m = std::min({mu_u, mu_p, mu_e});
        if      (mu_u==m) winner="Unpooled";
        else if (mu_p==m) winner="Pooled";
        else              winner="EB";

        out << std::fixed << std::setprecision(4)
            << p    << ',' << mu_u << ',' << se_u << ','
            << mu_p << ',' << se_p << ','
            << mu_e << ',' << se_e << ',' << winner << '\n';

        std::cout << "p=" << std::fixed << std::setprecision(2) << p
                  << "  U="  << std::setprecision(3) << mu_u
                  << "  P="  << mu_p
                  << "  EB=" << mu_e
                  << "  winner=" << winner << "\n";
    }

    out.close();
    std::cout << "\nDone! Results written to graphs_ti/highn.csv\n";
    return 0;
}
