#include <algorithm>
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

// Returns average cumulative regret over participants at each t.
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

    // Convert instantaneous regrets to mean cumulative regret trajectory.
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

int main() {
    // Mirrors the notebook smoke-test defaults.
    const int runs = 500;
    const int n = 100;
    const int arms = 2;
    const int T = 200;

    const Vec mu_a = {0.0, 1.0}; // context 0, arm 0 and arm 1
    const Vec tau_a = {0.2, 0.2}; // context 0, arm 0, arm 1
    const Vec theta_ctx_mu = {2, 0}; // added in context 1, arm 0 and arm 1
    const Vec theta_ctx_tau = {0.2, 0.2}; // added in context 1, arm 0 and arm 1
    const Vec sigma_a = {0.5, 0.5};
    const double p_one = 0.2;

    const Vec mu0 = {0.0, 0.0};
    const Vec tau0 = {1.0, 1.0};

    std::mt19937_64 seeder(20260308ULL);

    Vec mean_ts(T, 0.0), mean_pooled(T, 0.0), mean_eb(T, 0.0);

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

        for (int t = 0; t < T; ++t) {
            mean_ts[t] += ts[t];
            mean_pooled[t] += pts[t];
            mean_eb[t] += eb[t];
        }
    }

    for (int t = 0; t < T; ++t) {
        mean_ts[t] /= static_cast<double>(runs);
        mean_pooled[t] /= static_cast<double>(runs);
        mean_eb[t] /= static_cast<double>(runs);
    }

    std::ofstream out("mean_regret_over_time.csv");
    out << "t,mean_regret_thompson,mean_regret_pooled,mean_regret_eb\n";
    for (int t = 0; t < T; ++t) {
        out << (t + 1) << ","
            << std::fixed << std::setprecision(6)
            << mean_ts[t] << ","
            << mean_pooled[t] << ","
            << mean_eb[t] << "\n";
    }
    out.close();

    std::cout << "Saved mean_regret_over_time.csv\n";
    return 0;
}
