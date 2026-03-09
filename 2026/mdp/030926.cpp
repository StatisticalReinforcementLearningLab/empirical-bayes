#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using Vec4 = std::array<double, 4>;
using Mat4 = std::array<std::array<double, 4>, 4>;


// ============================================================
// 1. Small linear algebra helpers for 4x4 matrices
// ============================================================

Mat4 zero_mat4() {
    Mat4 A{};
    for (auto &row : A) row.fill(0.0);
    return A;
}

Mat4 identity_mat4(double c = 1.0) {
    Mat4 A = zero_mat4();
    for (int i = 0; i < 4; ++i) A[i][i] = c;
    return A;
}

Vec4 zero_vec4() {
    return {0.0, 0.0, 0.0, 0.0};
}

Vec4 matvec(const Mat4 &A, const Vec4 &x) {
    Vec4 out = zero_vec4();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            out[i] += A[i][j] * x[j];
        }
    }
    return out;
}

double dot(const Vec4 &a, const Vec4 &b) {
    double s = 0.0;
    for (int i = 0; i < 4; ++i) s += a[i] * b[i];
    return s;
}

Mat4 outer(const Vec4 &x) {
    Mat4 A = zero_mat4();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            A[i][j] = x[i] * x[j];
        }
    }
    return A;
}

Mat4 add(const Mat4 &A, const Mat4 &B) {
    Mat4 C = zero_mat4();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

Vec4 add(const Vec4 &a, const Vec4 &b) {
    Vec4 c{};
    for (int i = 0; i < 4; ++i) c[i] = a[i] + b[i];
    return c;
}

Vec4 sub(const Vec4 &a, const Vec4 &b) {
    Vec4 c{};
    for (int i = 0; i < 4; ++i) c[i] = a[i] - b[i];
    return c;
}

Vec4 scale(const Vec4 &a, double c) {
    Vec4 b{};
    for (int i = 0; i < 4; ++i) b[i] = c * a[i];
    return b;
}

Mat4 scale(const Mat4 &A, double c) {
    Mat4 B = zero_mat4();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            B[i][j] = c * A[i][j];
        }
    }
    return B;
}

// Simple Gauss-Jordan inverse for 4x4 matrices.
Mat4 inverse4(const Mat4 &A) {
    double aug[4][8] = {};

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) aug[i][j] = A[i][j];
        for (int j = 0; j < 4; ++j) aug[i][4 + j] = (i == j ? 1.0 : 0.0);
    }

    for (int col = 0; col < 4; ++col) {
        int pivot = col;
        for (int r = col + 1; r < 4; ++r) {
            if (std::fabs(aug[r][col]) > std::fabs(aug[pivot][col])) pivot = r;
        }
        if (std::fabs(aug[pivot][col]) < 1e-12) {
            throw std::runtime_error("Matrix inversion failed: near-singular matrix.");
        }
        if (pivot != col) {
            for (int j = 0; j < 8; ++j) std::swap(aug[col][j], aug[pivot][j]);
        }

        double piv = aug[col][col];
        for (int j = 0; j < 8; ++j) aug[col][j] /= piv;

        for (int r = 0; r < 4; ++r) {
            if (r == col) continue;
            double factor = aug[r][col];
            for (int j = 0; j < 8; ++j) aug[r][j] -= factor * aug[col][j];
        }
    }

    Mat4 inv = zero_mat4();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            inv[i][j] = aug[i][4 + j];
        }
    }
    return inv;
}

Mat4 cholesky4(const Mat4 &A) {
    Mat4 L = zero_mat4();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = A[i][j];
            for (int k = 0; k < j; ++k) sum -= L[i][k] * L[j][k];

            if (i == j) {
                if (sum < 1e-12) sum = 1e-12;
                L[i][j] = std::sqrt(sum);
            } else {
                L[i][j] = sum / L[j][j];
            }
        }
    }
    return L;
}

Vec4 mvn_sample(std::mt19937_64 &rng, const Vec4 &mu, const Mat4 &Sigma) {
    static std::normal_distribution<double> zdist(0.0, 1.0);

    Mat4 L = cholesky4(Sigma);
    Vec4 z{};
    for (int i = 0; i < 4; ++i) z[i] = zdist(rng);

    Vec4 out = mu;
    for (int i = 0; i < 4; ++i) {
        for (int k = 0; k <= i; ++k) {
            out[i] += L[i][k] * z[k];
        }
    }
    return out;
}


// ============================================================
// 2. Environment
// ============================================================

struct SimpleLinearMDP {
    int n;
    Vec4 mu_theta;
    Mat4 Sigma_theta;
    double alpha, beta, gamma;
    double sigma_s, sigma_r;
    double mu_s0, sigma_s0;

    std::mt19937_64 rng;
    std::vector<Vec4> theta;
    std::vector<double> s;

    SimpleLinearMDP(
        int n_,
        const Vec4 &mu_theta_,
        const Mat4 &Sigma_theta_,
        double alpha_,
        double beta_,
        double gamma_,
        double sigma_s_,
        double sigma_r_,
        double mu_s0_,
        double sigma_s0_,
        uint64_t seed
    )
        : n(n_),
          mu_theta(mu_theta_),
          Sigma_theta(Sigma_theta_),
          alpha(alpha_),
          beta(beta_),
          gamma(gamma_),
          sigma_s(sigma_s_),
          sigma_r(sigma_r_),
          mu_s0(mu_s0_),
          sigma_s0(sigma_s0_),
          rng(seed),
          theta(n_),
          s(n_, 0.0) {
        for (int i = 0; i < n; ++i) {
            theta[i] = mvn_sample(rng, mu_theta, Sigma_theta);
        }
    }

    std::vector<double> reset() {
        std::normal_distribution<double> dist(mu_s0, sigma_s0);
        for (int i = 0; i < n; ++i) s[i] = dist(rng);
        return s;
    }

    static Vec4 feature_vector(double s, int a) {
        return {1.0, s, static_cast<double>(a), s * static_cast<double>(a)};
    }

    std::pair<std::vector<double>, std::vector<double>> step(const std::vector<int> &a) {
        std::vector<double> reward(n), next_state(n);

        std::normal_distribution<double> rnoise(0.0, sigma_r);
        std::normal_distribution<double> snoise(0.0, sigma_s);

        for (int i = 0; i < n; ++i) {
            Vec4 x = feature_vector(s[i], a[i]);

            double reward_mean = dot(theta[i], x);
            reward[i] = reward_mean + rnoise(rng);

            double next_state_mean = alpha + beta * s[i] + gamma * static_cast<double>(a[i]);
            next_state[i] = next_state_mean + snoise(rng);
        }

        s = next_state;
        return {reward, next_state};
    }
};


// ============================================================
// 3. Shared helpers
// ============================================================

Vec4 feature_vector(double s, int a) {
    return {1.0, s, static_cast<double>(a), s * static_cast<double>(a)};
}

std::pair<Vec4, Mat4> posterior_from_precision(const Mat4 &Lambda, const Vec4 &b) {
    Mat4 Sigma = inverse4(Lambda);
    Vec4 mu = matvec(Sigma, b);
    return {mu, Sigma};
}


// ============================================================
// 4. Unpooled Thompson Sampling
// ============================================================

struct UnpooledTS {
    int n;
    double lambda_prior;
    std::mt19937_64 rng;
    std::vector<Mat4> Lambda;
    std::vector<Vec4> b;

    UnpooledTS(int n_, double lambda_prior_, uint64_t seed)
        : n(n_),
          lambda_prior(lambda_prior_),
          rng(seed),
          Lambda(n_, identity_mat4(lambda_prior_)),
          b(n_, zero_vec4()) {}

    std::vector<int> select_actions(const std::vector<double> &s) {
        std::vector<int> a(n, 0);

        for (int i = 0; i < n; ++i) {
            auto [mu_i, Sigma_i] = posterior_from_precision(Lambda[i], b[i]);
            Vec4 theta_tilde = mvn_sample(rng, mu_i, Sigma_i);

            double r0 = dot(feature_vector(s[i], 0), theta_tilde);
            double r1 = dot(feature_vector(s[i], 1), theta_tilde);
            a[i] = (r1 > r0) ? 1 : 0;
        }
        return a;
    }

    void update(const std::vector<double> &s, const std::vector<int> &a, const std::vector<double> &r) {
        for (int i = 0; i < n; ++i) {
            Vec4 x = feature_vector(s[i], a[i]);
            Lambda[i] = add(Lambda[i], outer(x));
            b[i] = add(b[i], scale(x, r[i]));
        }
    }
};


// ============================================================
// 5. Pooled Thompson Sampling
// ============================================================

struct PooledTS {
    int n;
    double lambda_prior;
    std::mt19937_64 rng;
    Mat4 Lambda;
    Vec4 b;

    PooledTS(int n_, double lambda_prior_, uint64_t seed)
        : n(n_),
          lambda_prior(lambda_prior_),
          rng(seed),
          Lambda(identity_mat4(lambda_prior_)),
          b(zero_vec4()) {}

    std::vector<int> select_actions(const std::vector<double> &s) {
        auto [mu, Sigma] = posterior_from_precision(Lambda, b);
        Vec4 theta_tilde = mvn_sample(rng, mu, Sigma);

        std::vector<int> a(n, 0);
        for (int i = 0; i < n; ++i) {
            double r0 = dot(feature_vector(s[i], 0), theta_tilde);
            double r1 = dot(feature_vector(s[i], 1), theta_tilde);
            a[i] = (r1 > r0) ? 1 : 0;
        }
        return a;
    }

    void update(const std::vector<double> &s, const std::vector<int> &a, const std::vector<double> &r) {
        for (int i = 0; i < n; ++i) {
            Vec4 x = feature_vector(s[i], a[i]);
            Lambda = add(Lambda, outer(x));
            b = add(b, scale(x, r[i]));
        }
    }
};


// ============================================================
// 6. Empirical Bayes Thompson Sampling
// ============================================================

struct EBTS {
    int n;
    double lambda_prior;
    std::mt19937_64 rng;

    std::vector<Mat4> Lambda_local;
    std::vector<Vec4> b_local;

    Vec4 mu_pop;
    Vec4 tau2_pop;

    EBTS(int n_, double lambda_prior_, uint64_t seed)
        : n(n_),
          lambda_prior(lambda_prior_),
          rng(seed),
          Lambda_local(n_, identity_mat4(lambda_prior_)),
          b_local(n_, zero_vec4()),
          mu_pop{0.0, 0.0, 0.0, 0.0},
          tau2_pop{1.0, 1.0, 1.0, 1.0} {}

    void estimate_population_hyperparams() {
        std::vector<Vec4> local_means(n);
        std::vector<Vec4> local_vars(n);

        for (int i = 0; i < n; ++i) {
            auto [mu_i, Sigma_i] = posterior_from_precision(Lambda_local[i], b_local[i]);
            local_means[i] = mu_i;
            for (int j = 0; j < 4; ++j) local_vars[i][j] = Sigma_i[j][j];
        }

        for (int j = 0; j < 4; ++j) {
            double num = 0.0, den = 0.0;
            for (int i = 0; i < n; ++i) {
                double w = 1.0 / std::max(local_vars[i][j], 1e-8);
                num += w * local_means[i][j];
                den += w;
            }
            mu_pop[j] = num / den;

            double centered2 = 0.0;
            double avg_within = 0.0;
            for (int i = 0; i < n; ++i) {
                double d = local_means[i][j] - mu_pop[j];
                centered2 += d * d;
                avg_within += local_vars[i][j];
            }
            centered2 /= static_cast<double>(n);
            avg_within /= static_cast<double>(n);
            tau2_pop[j] = std::max(centered2 - avg_within, 1e-6);
        }
    }

    std::pair<Vec4, Mat4> posterior_for_person(int i) {
        auto [mu_local, Sigma_local] = posterior_from_precision(Lambda_local[i], b_local[i]);

        Mat4 Sigma_pop_inv = zero_mat4();
        for (int j = 0; j < 4; ++j) Sigma_pop_inv[j][j] = 1.0 / std::max(tau2_pop[j], 1e-8);

        Mat4 Sigma_local_inv = inverse4(Sigma_local);
        Mat4 Sigma_post = inverse4(add(Sigma_local_inv, Sigma_pop_inv));

        Vec4 term1 = matvec(Sigma_local_inv, mu_local);
        Vec4 term2 = matvec(Sigma_pop_inv, mu_pop);
        Vec4 mu_post = matvec(Sigma_post, add(term1, term2));

        return {mu_post, Sigma_post};
    }

    std::vector<int> select_actions(const std::vector<double> &s) {
        estimate_population_hyperparams();

        std::vector<int> a(n, 0);
        for (int i = 0; i < n; ++i) {
            auto [mu_post, Sigma_post] = posterior_for_person(i);
            Vec4 theta_tilde = mvn_sample(rng, mu_post, Sigma_post);

            double r0 = dot(feature_vector(s[i], 0), theta_tilde);
            double r1 = dot(feature_vector(s[i], 1), theta_tilde);
            a[i] = (r1 > r0) ? 1 : 0;
        }
        return a;
    }

    void update(const std::vector<double> &s, const std::vector<int> &a, const std::vector<double> &r) {
        for (int i = 0; i < n; ++i) {
            Vec4 x = feature_vector(s[i], a[i]);
            Lambda_local[i] = add(Lambda_local[i], outer(x));
            b_local[i] = add(b_local[i], scale(x, r[i]));
        }
    }
};


// ============================================================
// 7. Simulation runner
// ============================================================

template <typename Alg>
std::vector<double> run_algorithm_mean_regret_over_time(SimpleLinearMDP &env, Alg &alg, int T) {
    std::vector<double> s = env.reset();
    std::vector<double> mean_regret_over_time(T, 0.0);

    for (int t = 0; t < T; ++t) {
        std::vector<int> a = alg.select_actions(s);

        double mean_regret = 0.0;
        for (int i = 0; i < env.n; ++i) {
            double mu0 = dot(feature_vector(s[i], 0), env.theta[i]);
            double mu1 = dot(feature_vector(s[i], 1), env.theta[i]);
            double best_mu = std::max(mu0, mu1);
            double chosen_mu = (a[i] == 1) ? mu1 : mu0;
            mean_regret += (best_mu - chosen_mu);
        }
        mean_regret /= static_cast<double>(env.n);

        auto [r, s_next] = env.step(a);
        alg.update(s, a, r);

        mean_regret_over_time[t] = mean_regret;
        s = s_next;
    }

    return mean_regret_over_time;
}

std::vector<double> average_runs(const std::string &algo_name, int n_runs, int n, int T,
                                 const Vec4 &mu_theta, const Mat4 &Sigma_theta,
                                 double alpha, double beta, double gamma,
                                 double sigma_s, double sigma_r,
                                 double mu_s0, double sigma_s0,
                                 double lambda_prior) {
    std::vector<double> avg(T, 0.0);

    for (int run = 0; run < n_runs; ++run) {
        uint64_t env_seed = 123456 + run;
        uint64_t alg_seed = 987654 + run;

        SimpleLinearMDP env(
            n, mu_theta, Sigma_theta,
            alpha, beta, gamma,
            sigma_s, sigma_r,
            mu_s0, sigma_s0,
            env_seed
        );

        std::vector<double> one_run;

        if (algo_name == "unpooled_ts") {
            UnpooledTS alg(n, lambda_prior, alg_seed);
            one_run = run_algorithm_mean_regret_over_time(env, alg, T);
        } else if (algo_name == "pooled_ts") {
            PooledTS alg(n, lambda_prior, alg_seed);
            one_run = run_algorithm_mean_regret_over_time(env, alg, T);
        } else if (algo_name == "eb_ts") {
            EBTS alg(n, lambda_prior, alg_seed);
            one_run = run_algorithm_mean_regret_over_time(env, alg, T);
        } else {
            throw std::runtime_error("Unknown algorithm name.");
        }

        for (int t = 0; t < T; ++t) {
            avg[t] += one_run[t];
        }
    }

    for (int t = 0; t < T; ++t) avg[t] /= static_cast<double>(n_runs);
    return avg;
}

void write_csv(const std::string &filename,
               const std::vector<double> &unpooled,
               const std::vector<double> &pooled,
               const std::vector<double> &eb) {
    std::ofstream out(filename);
    out << "time,unpooled_ts_mean_regret,pooled_ts_mean_regret,eb_ts_mean_regret\n";
    out << std::fixed << std::setprecision(8);

    int T = static_cast<int>(unpooled.size());
    for (int t = 0; t < T; ++t) {
        out << t + 1 << ","
            << unpooled[t] << ","
            << pooled[t] << ","
            << eb[t] << "\n";
    }
    out.close();
}


// ============================================================
// 8. Main
// ============================================================

int main() {
    const int n = 50;
    const int T = 40;
    const int n_runs = 500;
    const double lambda_prior = 1.0;

    Vec4 mu_theta = {0.0, 1.0, 0.5, -0.25};

    Mat4 Sigma_theta = zero_mat4();
    Sigma_theta[0][0] = 0.02;
    Sigma_theta[1][1] = 0.02;
    Sigma_theta[2][2] = 0.02;
    Sigma_theta[3][3] = 0.02;

    const double alpha = 0.0;
    const double beta = 0.8;
    const double gamma = 0.4;
    const double sigma_s = 0.5;
    const double sigma_r = 1.5;
    const double mu_s0 = 0.0;
    const double sigma_s0 = 1.0;

    std::vector<double> unpooled = average_runs(
        "unpooled_ts", n_runs, n, T,
        mu_theta, Sigma_theta,
        alpha, beta, gamma,
        sigma_s, sigma_r,
        mu_s0, sigma_s0,
        lambda_prior
    );

    std::vector<double> pooled = average_runs(
        "pooled_ts", n_runs, n, T,
        mu_theta, Sigma_theta,
        alpha, beta, gamma,
        sigma_s, sigma_r,
        mu_s0, sigma_s0,
        lambda_prior
    );

    std::vector<double> eb = average_runs(
        "eb_ts", n_runs, n, T,
        mu_theta, Sigma_theta,
        alpha, beta, gamma,
        sigma_s, sigma_r,
        mu_s0, sigma_s0,
        lambda_prior
    );

    write_csv("mean_regret_over_time_500_runs.csv", unpooled, pooled, eb);

    std::cout << "Done. Wrote mean_regret_over_time_500_runs.csv\n";
    return 0;
}