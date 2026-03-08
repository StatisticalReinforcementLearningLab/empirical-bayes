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

Mat set_environment(int n, const Vec& mu_a, const Vec& tau_a, std::mt19937_64& rng) {
    Mat theta(n, Vec(2, 0.0));
    for (int a = 0; a < 2; ++a) {
        std::normal_distribution<double> d(mu_a[a], tau_a[a]);
        for (int i = 0; i < n; ++i) theta[i][a] = d(rng);
    }
    return theta;
}

double unpooled_ts(int T, int n, const Mat& theta, double sigma_r, Vec mu0, Vec tau0, std::mt19937_64& rng) {
    IMat N(n, IVec(2, 0));
    Mat S(n, Vec(2, 0.0));
    double sig2 = sigma_r * sigma_r;
    double total_regret = 0.0;

    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < n; ++i) {
            Vec post_mean(2), post_var(2);
            for (int a = 0; a < 2; ++a) {
                double prec_prior = 1.0 / (tau0[a] * tau0[a]);
                double prec_data = N[i][a] / sig2;
                double v = 1.0 / (prec_prior + prec_data);
                double m = v * (mu0[a] / (tau0[a] * tau0[a]) + S[i][a] / sig2);
                if (N[i][a] == 0) {
                    m = mu0[a];
                    v = tau0[a] * tau0[a];
                }
                post_mean[a] = m;
                post_var[a] = v;
            }

            int chosen = 0;
            double best = -INFINITY;
            for (int a = 0; a < 2; ++a) {
                std::normal_distribution<double> d(post_mean[a], std::sqrt(post_var[a]));
                double s = d(rng);
                if (s > best) {
                    best = s;
                    chosen = a;
                }
            }

            std::normal_distribution<double> rd(theta[i][chosen], sigma_r);
            double r = rd(rng);
            N[i][chosen]++;
            S[i][chosen] += r;

            double opt = std::max(theta[i][0], theta[i][1]);
            total_regret += (opt - theta[i][chosen]);
        }
    }

    return total_regret / static_cast<double>(n);
}

double pooled_ts(int T, int n, const Mat& theta, double sigma_r, Vec mu0, Vec tau0, std::mt19937_64& rng) {
    IVec N(2, 0);
    Vec S(2, 0.0);
    double sig2 = sigma_r * sigma_r;
    double total_regret = 0.0;

    for (int t = 0; t < T; ++t) {
        Vec post_mean(2), post_var(2);
        for (int a = 0; a < 2; ++a) {
            double prec_prior = 1.0 / (tau0[a] * tau0[a]);
            double prec_data = N[a] / sig2;
            double v = 1.0 / (prec_prior + prec_data);
            double m = v * (mu0[a] / (tau0[a] * tau0[a]) + S[a] / sig2);
            if (N[a] == 0) {
                m = mu0[a];
                v = tau0[a] * tau0[a];
            }
            post_mean[a] = m;
            post_var[a] = v;
        }

        int chosen = 0;
        double best = -INFINITY;
        for (int a = 0; a < 2; ++a) {
            std::normal_distribution<double> d(post_mean[a], std::sqrt(post_var[a]));
            double s = d(rng);
            if (s > best) {
                best = s;
                chosen = a;
            }
        }

        for (int i = 0; i < n; ++i) {
            std::normal_distribution<double> rd(theta[i][chosen], sigma_r);
            double r = rd(rng);
            S[chosen] += r;
            double opt = std::max(theta[i][0], theta[i][1]);
            total_regret += (opt - theta[i][chosen]);
        }
        N[chosen] += n;
    }

    return total_regret / static_cast<double>(n);
}

inline double normal_cdf(double z) {
    return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
}

double empirical_bayes(int T, int n, const Mat& theta, double sigma_r, Vec mu0, Vec tau0, std::mt19937_64& rng) {
    Mat m(n, Vec(2));
    Mat v(n, Vec(2));
    for (int i = 0; i < n; ++i) {
        m[i][0] = mu0[0];
        m[i][1] = mu0[1];
        v[i][0] = tau0[0] * tau0[0];
        v[i][1] = tau0[1] * tau0[1];
    }

    double sig2 = sigma_r * sigma_r;
    double total_regret = 0.0;

    for (int t = 0; t < T; ++t) {
        Vec mu_hat(2, 0.0), tau2_hat(2, 0.0);
        for (int a = 0; a < 2; ++a) {
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
            Vec m_eb(2), v_eb(2);
            for (int a = 0; a < 2; ++a) {
                double lam = (tau2_hat[a] > 0.0) ? (tau2_hat[a] / (tau2_hat[a] + v[i][a])) : 1.0;
                m_eb[a] = lam * m[i][a] + (1.0 - lam) * mu_hat[a];
                v_eb[a] = (tau2_hat[a] > 0.0) ? (v[i][a] * tau2_hat[a]) / (v[i][a] + tau2_hat[a]) : 0.0;
            }

            double diff = m_eb[1] - m_eb[0];
            double vdiff = std::max(1e-20, v_eb[1] + v_eb[0]);
            double p1 = normal_cdf(diff / std::sqrt(vdiff));
            std::uniform_real_distribution<double> u(0.0, 1.0);
            int chosen = (u(rng) < p1) ? 1 : 0;

            std::normal_distribution<double> rd(theta[i][chosen], sigma_r);
            double r = rd(rng);

            double prec = 1.0 / v[i][chosen] + 1.0 / sig2;
            double v_new = 1.0 / prec;
            double m_new = v_new * (m[i][chosen] / v[i][chosen] + r / sig2);
            m[i][chosen] = m_new;
            v[i][chosen] = v_new;

            double opt = std::max(theta[i][0], theta[i][1]);
            total_regret += (opt - theta[i][chosen]);
        }
    }

    return total_regret / static_cast<double>(n);
}

int main() {
    const int n = 50;
    const int T = 200;
    const int runs = 500;

    const double theta0_mu = 2.0;
    const double theta1_mu_base = 2.0;
    const double tau0 = 1.0;
    const double tau1 = 1.0;
    const double sigma_r = 3.0;

    const Vec prior_mu = {2.0, 2.0};
    const Vec prior_tau = {1.5, 1.5};

    std::ofstream out("030826_ratio_sweep_results.csv");
    out << "ratio,theta1_mu_context0,theta1_mu_context1,mean_unpooled,se_unpooled,mean_pooled,se_pooled,mean_eb,se_eb,winner\n";

    std::mt19937_64 seeder(123456789ULL);

    for (double ratio = 1.0; ratio <= 5.0 + 1e-9; ratio += 0.25) {
        Vec regrets_unpooled, regrets_pooled, regrets_eb;
        regrets_unpooled.reserve(runs);
        regrets_pooled.reserve(runs);
        regrets_eb.reserve(runs);

        double theta1_mu_context0 = theta1_mu_base;
        double theta1_mu_context1 = theta1_mu_base * ratio;

        // Context-blind simulators see one arm-level mean per arm, averaged over contexts.
        Vec env_mu = {
            theta0_mu,
            0.5 * (theta1_mu_context0 + theta1_mu_context1)
        };
        Vec env_tau = {tau0, tau1};

        for (int r = 0; r < runs; ++r) {
            std::mt19937_64 rng(seeder());
            Mat theta = set_environment(n, env_mu, env_tau, rng);
            regrets_unpooled.push_back(unpooled_ts(T, n, theta, sigma_r, prior_mu, prior_tau, rng));
            regrets_pooled.push_back(pooled_ts(T, n, theta, sigma_r, prior_mu, prior_tau, rng));
            regrets_eb.push_back(empirical_bayes(T, n, theta, sigma_r, prior_mu, prior_tau, rng));
        }

        Summary su = summarize(regrets_unpooled);
        Summary sp = summarize(regrets_pooled);
        Summary se = summarize(regrets_eb);

        std::string winner = "Unpooled";
        double best = su.mean;
        if (sp.mean < best) {
            best = sp.mean;
            winner = "Pooled";
        }
        if (se.mean < best) {
            winner = "EB";
        }

        out << std::fixed << std::setprecision(4)
            << ratio << ","
            << theta1_mu_context0 << ","
            << theta1_mu_context1 << ","
            << su.mean << "," << su.se << ","
            << sp.mean << "," << sp.se << ","
            << se.mean << "," << se.se << ","
            << winner << "\n";

        std::cout << "ratio=" << std::setprecision(2) << ratio
                  << " winner=" << winner << "\n";
    }

    std::cout << "Saved 030826_ratio_sweep_results.csv\n";
    return 0;
}
