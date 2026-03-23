#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;
using IVec = std::vector<int>;
using IMat = std::vector<IVec>;

inline double normal_cdf(double z) {
    return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
}

struct Summary {
    double mean = 0.0;
    double se = 0.0;
};

Summary summarize(const Vec& x) {
    Summary s;
    if (x.empty()) return s;

    s.mean = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
    if (x.size() == 1) return s;

    double ss = 0.0;
    for (double v : x) {
        double d = v - s.mean;
        ss += d * d;
    }
    double var = ss / static_cast<double>(x.size() - 1);
    s.se = std::sqrt(var / static_cast<double>(x.size()));
    return s;
}

struct Env {
    Mat theta0;
    Mat theta1;
    Vec sigma_a;
    Vec contexts;
};

Env set_environment_contextual(
    int n,
    int arms,
    const Vec& mu_a,
    const Vec& tau_a,
    const Vec& theta_ctx_mu,
    const Vec& theta_ctx_tau,
    const Vec& sigma_a,
    double p_one,
    std::mt19937_64& rng
) {
    Env env;
    env.theta0.assign(n, Vec(arms, 0.0));
    env.theta1.assign(n, Vec(arms, 0.0));
    env.sigma_a = sigma_a;
    env.contexts.assign(n, 0.0);

    for (int a = 0; a < arms; ++a) {
        std::normal_distribution<double> d0(mu_a[a], tau_a[a]);
        std::normal_distribution<double> d1(theta_ctx_mu[a], theta_ctx_tau[a]);
        for (int i = 0; i < n; ++i) {
            env.theta0[i][a] = d0(rng);
            env.theta1[i][a] = d1(rng);
        }
    }

    std::bernoulli_distribution bd(p_one);
    for (int i = 0; i < n; ++i) {
        env.contexts[i] = bd(rng) ? 1.0 : 0.0;
    }

    return env;
}

Mat build_theta(const Env& env) {
    int n = static_cast<int>(env.theta0.size());
    int arms = static_cast<int>(env.theta0[0].size());
    Mat theta(n, Vec(arms, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int a = 0; a < arms; ++a) {
            theta[i][a] = env.theta0[i][a] + env.theta1[i][a] * env.contexts[i];
        }
    }
    return theta;
}

Vec run_thompson(
    int T,
    int n,
    int arms,
    const Mat& theta,
    const Vec& sigma_a,
    const Vec& mu0,
    const Vec& tau0,
    std::mt19937_64& rng
) {
    IMat N(n, IVec(arms, 0));
    Mat S(n, Vec(arms, 0.0));
    Vec cum_regret_sum(T, 0.0);

    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < n; ++i) {
            int chosen = 0;
            double best_sample = -INFINITY;
            for (int a = 0; a < arms; ++a) {
                double sig2 = sigma_a[a] * sigma_a[a];
                double prec_prior = 1.0 / (tau0[a] * tau0[a]);
                double prec_data = static_cast<double>(N[i][a]) / sig2;
                double post_var = 1.0 / (prec_prior + prec_data);
                double post_mean = post_var * (mu0[a] / (tau0[a] * tau0[a]) + S[i][a] / sig2);
                if (N[i][a] == 0) {
                    post_mean = mu0[a];
                    post_var = tau0[a] * tau0[a];
                }
                std::normal_distribution<double> pd(post_mean, std::sqrt(post_var));
                double sample = pd(rng);
                if (sample > best_sample) {
                    best_sample = sample;
                    chosen = a;
                }
            }

            std::normal_distribution<double> rd(theta[i][chosen], sigma_a[chosen]);
            double reward = rd(rng);
            N[i][chosen] += 1;
            S[i][chosen] += reward;

            double best_theta = *std::max_element(theta[i].begin(), theta[i].end());
            double inst_regret = best_theta - theta[i][chosen];
            cum_regret_sum[t] += inst_regret;
        }
    }

    Vec cum(T, 0.0);
    double running = 0.0;
    for (int t = 0; t < T; ++t) {
        running += cum_regret_sum[t] / static_cast<double>(n);
        cum[t] = running;
    }
    return cum;
}

Vec run_pooled_thompson(
    int T,
    int n,
    int arms,
    const Mat& theta,
    const Vec& sigma_a,
    const Vec& mu0,
    const Vec& tau0,
    std::mt19937_64& rng
) {
    IVec N(arms, 0);
    Vec S(arms, 0.0);
    Vec cum_regret_sum(T, 0.0);

    for (int t = 0; t < T; ++t) {
        int chosen = 0;
        double best_sample = -INFINITY;
        for (int a = 0; a < arms; ++a) {
            double sig2 = sigma_a[a] * sigma_a[a];
            double prec_prior = 1.0 / (tau0[a] * tau0[a]);
            double prec_data = static_cast<double>(N[a]) / sig2;
            double post_var = 1.0 / (prec_prior + prec_data);
            double post_mean = post_var * (mu0[a] / (tau0[a] * tau0[a]) + S[a] / sig2);
            if (N[a] == 0) {
                post_mean = mu0[a];
                post_var = tau0[a] * tau0[a];
            }
            std::normal_distribution<double> pd(post_mean, std::sqrt(post_var));
            double sample = pd(rng);
            if (sample > best_sample) {
                best_sample = sample;
                chosen = a;
            }
        }

        for (int i = 0; i < n; ++i) {
            std::normal_distribution<double> rd(theta[i][chosen], sigma_a[chosen]);
            double reward = rd(rng);
            S[chosen] += reward;

            double best_theta = *std::max_element(theta[i].begin(), theta[i].end());
            double inst_regret = best_theta - theta[i][chosen];
            cum_regret_sum[t] += inst_regret;
        }
        N[chosen] += n;
    }

    Vec cum(T, 0.0);
    double running = 0.0;
    for (int t = 0; t < T; ++t) {
        running += cum_regret_sum[t] / static_cast<double>(n);
        cum[t] = running;
    }
    return cum;
}

Vec run_empirical_bayes(
    int T,
    int n,
    int arms,
    const Mat& theta,
    const Vec& sigma_a,
    const Vec& mu0,
    const Vec& tau0,
    std::mt19937_64& rng
) {
    Mat m(n, Vec(arms, 0.0));
    Mat v(n, Vec(arms, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int a = 0; a < arms; ++a) {
            m[i][a] = mu0[a];
            v[i][a] = tau0[a] * tau0[a];
        }
    }

    Vec cum_regret_sum(T, 0.0);

    for (int t = 0; t < T; ++t) {
        Vec mu_hat(arms, 0.0), tau2_hat(arms, 0.0);
        for (int a = 0; a < arms; ++a) {
            double mean_m = 0.0;
            double mean_v = 0.0;
            for (int i = 0; i < n; ++i) {
                mean_m += m[i][a];
                mean_v += v[i][a];
            }
            mean_m /= static_cast<double>(n);
            mean_v /= static_cast<double>(n);

            double var_m = 0.0;
            for (int i = 0; i < n; ++i) {
                double d = m[i][a] - mean_m;
                var_m += d * d;
            }
            var_m = (n > 1) ? var_m / static_cast<double>(n - 1) : 0.0;

            mu_hat[a] = mean_m;
            tau2_hat[a] = std::max(0.0, var_m - mean_v);
        }

        for (int i = 0; i < n; ++i) {
            Vec m_eb(arms, 0.0), v_eb(arms, 0.0);
            for (int a = 0; a < arms; ++a) {
                double lam = (tau2_hat[a] > 0.0) ? (tau2_hat[a] / (tau2_hat[a] + v[i][a])) : 1.0;
                m_eb[a] = lam * m[i][a] + (1.0 - lam) * mu_hat[a];
                v_eb[a] = (tau2_hat[a] > 0.0) ? (v[i][a] * tau2_hat[a]) / (v[i][a] + tau2_hat[a]) : 0.0;
            }

            int chosen = 0;
            if (arms == 2) {
                double diff = m_eb[1] - m_eb[0];
                double var_diff = std::max(1e-20, v_eb[1] + v_eb[0]);
                double p1 = normal_cdf(diff / std::sqrt(var_diff));
                std::uniform_real_distribution<double> ud(0.0, 1.0);
                chosen = (ud(rng) < p1) ? 1 : 0;
            } else {
                double best_sample = -INFINITY;
                for (int a = 0; a < arms; ++a) {
                    std::normal_distribution<double> pd(m_eb[a], std::sqrt(std::max(0.0, v_eb[a])));
                    double sample = pd(rng);
                    if (sample > best_sample) {
                        best_sample = sample;
                        chosen = a;
                    }
                }
            }

            std::normal_distribution<double> rd(theta[i][chosen], sigma_a[chosen]);
            double reward = rd(rng);

            double sig2 = sigma_a[chosen] * sigma_a[chosen];
            double prec = 1.0 / v[i][chosen] + 1.0 / sig2;
            double v_new = 1.0 / prec;
            double m_new = v_new * (m[i][chosen] / v[i][chosen] + reward / sig2);
            m[i][chosen] = m_new;
            v[i][chosen] = v_new;

            double best_theta = *std::max_element(theta[i].begin(), theta[i].end());
            double inst_regret = best_theta - theta[i][chosen];
            cum_regret_sum[t] += inst_regret;
        }
    }

    Vec cum(T, 0.0);
    double running = 0.0;
    for (int t = 0; t < T; ++t) {
        running += cum_regret_sum[t] / static_cast<double>(n);
        cum[t] = running;
    }
    return cum;
}

int main(int argc, char* argv[]) {
    // Keep defaults aligned with mean_regret_over_time.cpp.
    const int runs = 500;
    const int n = 100;
    const int arms = 2;
    const int T = 200;

    const Vec mu_a = {0.0, 1.0};
    const Vec tau_a = {0.2, 0.2};
    const Vec theta_ctx_tau = {0.2, 0.2};
    const Vec sigma_a = {0.5, 0.5};
    double p_one = 0.2;
    std::string out_csv = "final_regret_vs_distance_ratio.csv";

    if (argc >= 2) {
        p_one = std::atof(argv[1]);
    }
    if (argc >= 3) {
        out_csv = argv[2];
    }

    const Vec mu0 = {0.0, 0.0};
    const Vec tau0 = {1.0, 1.0};

    const double base_gap = std::abs(mu_a[1] - mu_a[0]);

    std::ofstream out(out_csv);
    out << "distance_ratio,theta_ctx_mu_arm0,mean_final_regret_thompson,se_final_regret_thompson,"
        << "mean_final_regret_pooled,se_final_regret_pooled,mean_final_regret_eb,se_final_regret_eb,winner\n";

    std::mt19937_64 seeder(20260308ULL);

    const Vec ratios = {0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0};

    for (double ratio : ratios) {
        const double theta_ctx_effect_arm0 = ratio * base_gap;
        const Vec theta_ctx_mu = {theta_ctx_effect_arm0, 0.0};
        const double distance_ratio = (base_gap > 0.0) ? (theta_ctx_effect_arm0 / base_gap) : 0.0;

        Vec final_ts;
        Vec final_pooled;
        Vec final_eb;
        final_ts.reserve(runs);
        final_pooled.reserve(runs);
        final_eb.reserve(runs);

        for (int run = 0; run < runs; ++run) {
            std::mt19937_64 rng_env(seeder());
            std::mt19937_64 rng_ts(seeder());
            std::mt19937_64 rng_pts(seeder());
            std::mt19937_64 rng_eb(seeder());

            Env env = set_environment_contextual(
                n, arms, mu_a, tau_a, theta_ctx_mu, theta_ctx_tau, sigma_a, p_one, rng_env
            );
            Mat theta = build_theta(env);

            Vec ts = run_thompson(T, n, arms, theta, sigma_a, mu0, tau0, rng_ts);
            Vec pts = run_pooled_thompson(T, n, arms, theta, sigma_a, mu0, tau0, rng_pts);
            Vec eb = run_empirical_bayes(T, n, arms, theta, sigma_a, mu0, tau0, rng_eb);

            final_ts.push_back(ts.back());
            final_pooled.push_back(pts.back());
            final_eb.push_back(eb.back());
        }

        Summary s_ts = summarize(final_ts);
        Summary s_pooled = summarize(final_pooled);
        Summary s_eb = summarize(final_eb);

        std::string winner = "Unpooled TS";
        double best = s_ts.mean;
        if (s_pooled.mean < best) {
            best = s_pooled.mean;
            winner = "Pooled TS";
        }
        if (s_eb.mean < best) {
            winner = "EB";
        }

        out << std::fixed << std::setprecision(6)
            << distance_ratio << ","
            << theta_ctx_effect_arm0 << ","
            << s_ts.mean << "," << s_ts.se << ","
            << s_pooled.mean << "," << s_pooled.se << ","
            << s_eb.mean << "," << s_eb.se << ","
            << winner << "\n";

        std::cout << "theta_ctx_mu_arm0=" << theta_ctx_effect_arm0
                  << " ratio=" << distance_ratio
                  << " winner=" << winner << "\n";
    }

    std::cout << "Saved " << out_csv << " (p_one=" << p_one << ")\n";
    return 0;
}