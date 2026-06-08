#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <numeric>
#include <iomanip>

using Vec  = std::vector<double>;
using Mat  = std::vector<Vec>;
using Ten  = std::vector<Mat>;
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
inline M2 sub2(const M2& A, const M2& B){ return {A.a-B.a,A.b-B.b,A.c-B.c,A.d-B.d}; }
inline M2 scale2(const M2& A, double s){ return {A.a*s,A.b*s,A.c*s,A.d*s}; }
inline Vec matvec2(const M2& M, const Vec& x){ return {M.a*x[0]+M.b*x[1], M.c*x[0]+M.d*x[1]}; }
inline double quad2(const Vec& x, const M2& M){
    return x[0]*(M.a*x[0]+M.b*x[1]) + x[1]*(M.c*x[0]+M.d*x[1]);
}
inline M2 outer2(const Vec& x){ return {x[0]*x[0], x[0]*x[1], x[1]*x[0], x[1]*x[1]}; }

inline M2 project_psd(const M2& M, double floor=0.0) {
    double a=M.a, d=M.d, b=0.5*(M.b+M.c);
    double tr = a+d;
    double det = a*d - b*b;
    double disc = std::sqrt(std::max(0.25*tr*tr - det, 0.0));
    double l1 = std::max(0.5*tr + disc, floor);
    double l2 = std::max(0.5*tr - disc, floor);
    if (std::abs(b) < 1e-14)
        return { std::max(a,floor), 0.0, 0.0, std::max(d,floor) };
    double v1x = l1-d, v1y = b;
    double n1 = std::sqrt(v1x*v1x+v1y*v1y); v1x/=n1; v1y/=n1;
    double v2x = -v1y, v2y = v1x;
    return { l1*v1x*v1x+l2*v2x*v2x,
             l1*v1x*v1y+l2*v2x*v2y,
             l1*v1x*v1y+l2*v2x*v2y,
             l1*v1y*v1y+l2*v2y*v2y };
}

inline Vec sample_mvn2(const Vec& mu, const M2& S, std::mt19937_64& rng) {
    double a = std::sqrt(std::max(S.a, 0.0));
    double l21 = (a>1e-14) ? S.c/a : 0.0;
    double l22 = std::sqrt(std::max(S.d - l21*l21, 0.0));
    std::normal_distribution<double> nd(0.0,1.0);
    double z1=nd(rng), z2=nd(rng);
    return { mu[0]+a*z1, mu[1]+l21*z1+l22*z2 };
}

inline double normal_cdf(double z) {
    return 0.5*(1.0+std::erf(z/std::sqrt(2.0)));
}

// ---------------- Environment ----------------
struct Env {
    std::vector<std::vector<Vec>> theta; // n x 2 x 2
    IMat context;                         // n x T
};

Env set_environment(int n, int T, const std::vector<Vec>& mu_a,
                    const std::vector<M2>& Sigma_a, double p_context,
                    std::mt19937_64& rng) {
    Env env;
    env.theta.assign(n, std::vector<Vec>(2, Vec(2,0.0)));
    for (int a=0; a<2; ++a)
        for (int i=0; i<n; ++i)
            env.theta[i][a] = sample_mvn2(mu_a[a], Sigma_a[a], rng);
    env.context.assign(n, IVec(T,0));
    std::bernoulli_distribution bd(p_context);
    for (int i=0; i<n; ++i)
        for (int t=0; t<T; ++t)
            env.context[i][t] = bd(rng) ? 1 : 0;
    return env;
}

inline double expected_reward(const Env& env, int i, int t, int a) {
    int c = env.context[i][t];
    return env.theta[i][a][0] + env.theta[i][a][1]*c;
}
inline double best_expected(const Env& env, int i, int t) {
    return std::max(expected_reward(env,i,t,0), expected_reward(env,i,t,1));
}

// ---------------- Ridge state ----------------
struct RidgeState {
    M2 XtX;
    Vec Xty;
    Vec r_hist;
    std::vector<Vec> x_hist;
    int n_obs;
};

inline void init_state(RidgeState& s) {
    s.XtX = {0,0,0,0};
    s.Xty = {0.0,0.0};
    s.n_obs = 0;
}
inline void update_state(RidgeState& s, const Vec& x, double r) {
    s.XtX.a += x[0]*x[0]; s.XtX.b += x[0]*x[1];
    s.XtX.c += x[1]*x[0]; s.XtX.d += x[1]*x[1];
    s.Xty[0] += x[0]*r;   s.Xty[1] += x[1]*r;
    s.x_hist.push_back(x);
    s.r_hist.push_back(r);
    s.n_obs++;
}

inline void ridge_estimate(const RidgeState& s, double lam, const Vec& mu_prior,
                            Vec& theta_hat, M2& Sigma_hat) {
    M2 G = { s.XtX.a+lam, s.XtX.b, s.XtX.c, s.XtX.d+lam };
    M2 Ginv = inv2(G);
    theta_hat = matvec2(Ginv, s.Xty);
    if (s.n_obs == 0) theta_hat = mu_prior;
    double sigma2 = 1.0;
    if (s.n_obs > 2) {
        double rss = 0.0;
        for (int j=0; j<s.n_obs; ++j) {
            double pred = s.x_hist[j][0]*theta_hat[0]+s.x_hist[j][1]*theta_hat[1];
            double e = s.r_hist[j]-pred;
            rss += e*e;
        }
        sigma2 = std::max(rss/(s.n_obs-2), 1e-8);
    }
    Sigma_hat = scale2(Ginv, sigma2);
}

// ---------------- Unpooled Contextual TS ----------------
double unpooled_ts(int T, int n, const Env& env, double sigma_r, double lam,
                   const Vec& mu0_prior, std::mt19937_64& rng) {
    std::vector<std::vector<RidgeState>> state(n, std::vector<RidgeState>(2));
    for (int i=0; i<n; ++i) for (int a=0; a<2; ++a) init_state(state[i][a]);

    double total_regret = 0.0;
    std::uniform_int_distribution<int> ud(0,1);
    std::normal_distribution<double>   nd(0.0,1.0);

    for (int t=0; t<T; ++t) {
        for (int i=0; i<n; ++i) {
            Vec x = {1.0, (double)env.context[i][t]};
            int a_sel;
            if (state[i][0].n_obs == 0 || state[i][1].n_obs == 0) {
                a_sel = ud(rng);
            } else {
                Vec theta_hat[2]; M2 Sigma_hat[2];
                double samp_reward[2];
                for (int a=0; a<2; ++a) {
                    ridge_estimate(state[i][a], lam, mu0_prior,
                                   theta_hat[a], Sigma_hat[a]);
                    Vec th_draw = sample_mvn2(theta_hat[a], Sigma_hat[a], rng);
                    samp_reward[a] = x[0]*th_draw[0]+x[1]*th_draw[1];
                }
                a_sel = (samp_reward[1] > samp_reward[0]) ? 1 : 0;
            }
            double mean_r = expected_reward(env,i,t,a_sel);
            double r      = mean_r + sigma_r*nd(rng);
            update_state(state[i][a_sel], x, r);
            total_regret += best_expected(env,i,t) - mean_r;
        }
    }
    return total_regret / n;
}

// ---------------- Pooled Contextual TS ----------------
double pooled_ts(int T, int n, const Env& env, double sigma_r, double lam,
                 const Vec& mu0_prior, std::mt19937_64& rng) {
    std::vector<RidgeState> state(2);
    for (int a=0; a<2; ++a) init_state(state[a]);

    double total_regret = 0.0;
    std::uniform_int_distribution<int> ud(0,1);
    std::normal_distribution<double>   nd(0.0,1.0);

    for (int t=0; t<T; ++t) {
        Vec theta_hat[2]; M2 Sigma_hat[2];
        bool ready[2];
        for (int a=0; a<2; ++a) {
            ridge_estimate(state[a], lam, mu0_prior, theta_hat[a], Sigma_hat[a]);
            ready[a] = state[a].n_obs > 0;
        }
        for (int i=0; i<n; ++i) {
            Vec x = {1.0, (double)env.context[i][t]};
            int a_sel;
            if (!ready[0] || !ready[1]) {
                a_sel = ud(rng);
            } else {
                double samp_reward[2];
                for (int a=0; a<2; ++a) {
                    Vec th_draw = sample_mvn2(theta_hat[a], Sigma_hat[a], rng);
                    samp_reward[a] = x[0]*th_draw[0]+x[1]*th_draw[1];
                }
                a_sel = (samp_reward[1] > samp_reward[0]) ? 1 : 0;
            }
            double mean_r = expected_reward(env,i,t,a_sel);
            double r      = mean_r + sigma_r*nd(rng);
            update_state(state[a_sel], x, r);
            total_regret += best_expected(env,i,t) - mean_r;
        }
    }
    return total_regret / n;
}

// ---------------- Empirical Bayes ----------------
double empirical_bayes(int T, int n, const Env& env, double sigma_r, double lam,
                       const Vec& mu0_prior, std::mt19937_64& rng) {
    std::vector<std::vector<RidgeState>> state(n, std::vector<RidgeState>(2));
    for (int i=0; i<n; ++i) for (int a=0; a<2; ++a) init_state(state[i][a]);

    double total_regret = 0.0;
    std::uniform_int_distribution<int>     ud(0,1);
    std::uniform_real_distribution<double> ur(0.0,1.0);
    std::normal_distribution<double>       nd(0.0,1.0);

    Vec mu_pop[2]    = {mu0_prior, mu0_prior};
    M2  Sigma_pop[2] = {{1,0,0,1},{1,0,0,1}};

    std::vector<std::vector<Vec>>  theta_hat(n, std::vector<Vec>(2, Vec(2,0.0)));
    std::vector<std::vector<M2>>   Sigma_hat(n, std::vector<M2> (2, {1,0,0,1}));
    std::vector<std::vector<bool>> has_data (n, std::vector<bool>(2, false));

    const int    MAX_ITER = 10;
    const double TOL      = 1e-5;

    for (int t=0; t<T; ++t) {
        for (int i=0; i<n; ++i) for (int a=0; a<2; ++a) {
            ridge_estimate(state[i][a], lam, mu0_prior,
                           theta_hat[i][a], Sigma_hat[i][a]);
            has_data[i][a] = state[i][a].n_obs > 0;
        }

        for (int a=0; a<2; ++a) {
            int nn = 0;
            for (int i=0; i<n; ++i) if (has_data[i][a]) nn++;
            if (nn==0) continue;
            Vec mu_cur    = mu_pop[a];
            M2  Sigma_cur = Sigma_pop[a];
            for (int iter=0; iter<MAX_ITER; ++iter) {
                Vec mu_new    = {0.0, 0.0};
                M2  Sigma_new = {0.0, 0.0, 0.0, 0.0};
                for (int i=0; i<n; ++i) {
                    if (!has_data[i][a]) continue;
                    M2  Sp_inv    = inv2(Sigma_cur);
                    M2  Si_inv    = inv2(Sigma_hat[i][a]);
                    M2  post_cov  = inv2(add2(Sp_inv, Si_inv));
                    Vec rhs       = matvec2(Sp_inv, mu_cur);
                    Vec tmp       = matvec2(Si_inv, theta_hat[i][a]);
                    rhs[0]+=tmp[0]; rhs[1]+=tmp[1];
                    Vec post_mean = matvec2(post_cov, rhs);
                    mu_new[0] += post_mean[0];
                    mu_new[1] += post_mean[1];
                    Sigma_new = add2(Sigma_new, add2(post_cov, outer2(post_mean)));
                }
                mu_new[0] /= nn; mu_new[1] /= nn;
                Sigma_new = sub2(scale2(Sigma_new, 1.0/nn), outer2(mu_new));
                Sigma_new = project_psd(Sigma_new, 1e-8);
                double delta = std::abs(mu_new[0]-mu_cur[0])
                             + std::abs(mu_new[1]-mu_cur[1])
                             + std::abs(Sigma_new.a-Sigma_cur.a)
                             + std::abs(Sigma_new.d-Sigma_cur.d);
                mu_cur = mu_new; Sigma_cur = Sigma_new;
                if (delta < TOL) break;
            }
            mu_pop[a] = mu_cur; Sigma_pop[a] = Sigma_cur;
        }

        M2 Sp_inv[2] = { inv2(Sigma_pop[0]), inv2(Sigma_pop[1]) };
        for (int i=0; i<n; ++i) {
            Vec x = {1.0, (double)env.context[i][t]};
            int a_sel;
            if (!has_data[i][0] || !has_data[i][1]) {
                a_sel = ud(rng);
            } else {
                double reward_mean[2], reward_var[2];
                for (int a=0; a<2; ++a) {
                    M2  Si_inv    = inv2(Sigma_hat[i][a]);
                    M2  post_cov  = inv2(add2(Sp_inv[a], Si_inv));
                    Vec rhs       = matvec2(Sp_inv[a], mu_pop[a]);
                    Vec tmp       = matvec2(Si_inv, theta_hat[i][a]);
                    rhs[0]+=tmp[0]; rhs[1]+=tmp[1];
                    Vec post_mean = matvec2(post_cov, rhs);
                    reward_mean[a] = x[0]*post_mean[0]+x[1]*post_mean[1];
                    reward_var[a]  = quad2(x, post_cov);
                }
                double diff = reward_mean[1]-reward_mean[0];
                double sd   = std::sqrt(std::max(reward_var[0]+reward_var[1], 1e-20));
                a_sel = (ur(rng) < normal_cdf(diff/sd)) ? 1 : 0;
            }
            double mean_r = expected_reward(env,i,t,a_sel);
            double r      = mean_r + sigma_r*nd(rng);
            update_state(state[i][a_sel], x, r);
            total_regret += best_expected(env,i,t) - mean_r;
        }
    }
    return total_regret / n;
}

// ---------------- Main ----------------
int main() {
    int n = 100; 
    // int T = 50,
    int runs = 300;
    double sigma_r = 0.5; 
    double lam = 1e-6;
    double p_context = 0.5;

    std::vector<Vec> mu_a = {
        {0.0,  0.0},
        {0.07, 0.04}
    };
    std::vector<M2> Sigma_a = {
        {0.50, 0.0, 0.0, 0.50},
        {0.50, 0.0, 0.0, 0.50}
    };
    Vec mu0_prior = {0.0, 0.0};

    // Vec p_values;
    // for (double p=0.10; p<=0.50+1e-9; p+=0.05) p_values.push_back(p);
    // int np = p_values.size();

    // SWEEPING T! 
    IVec T_values;
    for (int Tv=1; Tv<=25; Tv+=1) T_values.push_back(Tv);
    int nT = T_values.size();

    // std::ofstream out("testing_env_betina/env_1_axes_low_T.csv");
    // out << "p_context,mean_unpooled,se_unpooled,mean_pooled,se_pooled,"
    //        "mean_eb,se_eb,winner\n";

    // std::cout << "Sweeping " << np << " p values...\n";
    std::ofstream out("T_sweep/T_sweep_conj3_a.csv");
    out << "T,mean_unpooled,se_unpooled,mean_pooled,se_pooled,"
       "mean_eb,se_eb,winner\n";

    std::cout << "Sweeping " << nT << " T values (time-variant context)...\n";
    

    // for (int pi=0; pi<np; ++pi) {
    for (int ti=0; ti<nT; ++ti) {
        // double p = p_values[pi];
        // double sum_u=0, sum_p=0, sum_e=0;
        // double ssq_u=0, ssq_p=0, ssq_e=0;
        double T = T_values[ti];
        double sum_u=0, sum_p=0, sum_e=0;
        double ssq_u=0, ssq_p=0, ssq_e=0;

        for (int r=0; r<runs; ++r) {
            std::mt19937_64 rng_env(r + 1000 + ti*10000);
            Env env = set_environment(n, T, mu_a, Sigma_a, p_context, rng_env);

            std::mt19937_64 rng_u(r + 2000 + ti*10000);
            double reg_u = unpooled_ts(T, n, env, sigma_r, lam, mu0_prior, rng_u);

            std::mt19937_64 rng_p(r + 3000 + ti*10000);
            double reg_p = pooled_ts(T, n, env, sigma_r, lam, mu0_prior, rng_p);

            std::mt19937_64 rng_e(r + 4000 + ti*10000);
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
        if      (mu_u == m) winner="Unpooled";
        else if (mu_p == m) winner="Pooled";
        else                winner="EB";

        out << std::fixed << std::setprecision(4)
            << T    << ',' << mu_u << ',' << se_u << ','
            << mu_p << ',' << se_p << ','
            << mu_e << ',' << se_e << ',' << winner << '\n';

        std::cout << "T=" << std::fixed << std::setprecision(2) << T
                  << "  U="  << std::setprecision(3) << mu_u
                  << "  P="  << mu_p
                  << "  EB=" << mu_e
                  << "  winner=" << winner << "\n";
    }
    
    out.close();
    std::cout << "\nDone! Results written to T_sweep/T_sweep_conj3_a.csv\n";
    return 0;
}