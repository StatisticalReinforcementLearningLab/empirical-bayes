#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <numeric>
#include <iomanip>

using Vec    = std::vector<double>;
using Mat    = std::vector<Vec>;
using IVec   = std::vector<int>;
using IMat   = std::vector<IVec>;

// sample from normal distribution vectorised
void normal_rng(double mean, double sd, Vec &out, std::mt19937_64 &rng) {
    std::normal_distribution<double> dist(mean, sd);
    for (auto &x : out) x = dist(rng);
}

// create environment: theta ~ Normal(mu_a, tau_a)
Mat set_environment(int n, int arms, const Vec &mu_a, const Vec &tau_a, std::mt19937_64 &rng) {
    Mat theta(n, Vec(arms));
    for (int a=0; a<arms; ++a) {
        std::normal_distribution<double> d(mu_a[a], tau_a[a]);
        for (int i=0; i<n; ++i) theta[i][a] = d(rng);
    }
    return theta;
}

// unpooled Thompson sampling
void thompson_sampling(int T, int n, int arms,
                       const Mat &theta, double sigma_r,
                       Vec mu0, Vec tau0,
                       IMat &actions, Mat &rewards,
                       std::mt19937_64 &rng) {
    actions.assign(n, IVec(T));
    rewards.assign(n, Vec(T));

    IMat N(n, IVec(arms,0));
    Mat S(n, Vec(arms,0.0));

    double sig2 = sigma_r*sigma_r;
    for (int t=0; t<T; ++t) {
        // posterior mean/var per (i,a)
        Mat post_mean(n, Vec(arms));
        Mat post_var(n, Vec(arms));
        for (int i=0; i<n; ++i) {
            for (int a=0; a<arms; ++a) {
                double prec_prior = 1.0/(tau0[a]*tau0[a]);
                double prec_data  = N[i][a] / sig2;
                double v = 1.0/(prec_data + prec_prior);
                double m = v*(mu0[a]/(tau0[a]*tau0[a]) + S[i][a]/sig2);
                if (N[i][a]==0) { m = mu0[a]; v = tau0[a]*tau0[a]; }
                post_mean[i][a] = m;
                post_var[i][a]  = v;
            }
        }
        // sample and choose
        for (int i=0;i<n;++i) {
            double best_val=-INFINITY;
            int best_a=0;
            for (int a=0;a<arms;++a) {
                std::normal_distribution<double> dist(post_mean[i][a], std::sqrt(post_var[i][a]));
                double sample = dist(rng);
                if (sample>best_val) { best_val=sample; best_a=a; }
            }
            actions[i][t]=best_a;
            // draw reward
            std::normal_distribution<double> rd(theta[i][best_a], sigma_r);
            double r = rd(rng);
            rewards[i][t]=r;
            N[i][best_a] += 1;
            S[i][best_a] += r;
        }
    }
}

// pooled Thompson sampling
void pooled_thompson_sampling(int T,int n,int arms,const Mat &theta,double sigma_r,
                              Vec mu0, Vec tau0,
                              IMat &actions, Mat &rewards,
                              std::mt19937_64 &rng) {
    actions.assign(n, IVec(T));
    rewards.assign(n, Vec(T));
    IVec N(arms,0);
    Vec S(arms,0.0);
    double sig2 = sigma_r*sigma_r;
    for(int t=0;t<T;++t){
        Vec post_mean(arms), post_var(arms);
        for(int a=0;a<arms;++a){
            double prec_prior = 1.0/(tau0[a]*tau0[a]);
            double prec_data  = N[a]/sig2;
            double v = 1.0/(prec_data+prec_prior);
            double m = v*(mu0[a]/(tau0[a]*tau0[a]) + S[a]/sig2);
            if(N[a]==0){m=mu0[a]; v=tau0[a]*tau0[a];}
            post_mean[a]=m; post_var[a]=v;
        }
        int best_a=0;
        double best_val=-INFINITY;
        for(int a=0;a<arms;++a){
            std::normal_distribution<double> dist(post_mean[a], std::sqrt(post_var[a]));
            double samp=dist(rng);
            if(samp>best_val){best_val=samp; best_a=a;}
        }
        for(int i=0;i<n;++i){
            actions[i][t]=best_a;
            std::normal_distribution<double> rd(theta[i][best_a], sigma_r);
            double r=rd(rng);
            rewards[i][t]=r;
        }
        N[best_a]+=n;
        double sumr=0;
        for(int i=0;i<n;++i) sumr+=rewards[i][t];
        S[best_a]+=sumr;
    }
}

// compute average regret curve (mean across participants) for one run
Vec cumulative_regret(const Mat &theta, const IMat &actions) {
    int n = actions.size();
    int T = actions[0].size();
    Vec out(T, 0.0);
    for (int i = 0; i < n; ++i) {
        double best = *std::max_element(theta[i].begin(), theta[i].end());
        double cum = 0.0;
        for (int t = 0; t < T; ++t) {
            double chosen = theta[i][ actions[i][t] ];
            cum += best - chosen;
            out[t] += cum;
        }
    }
    for (int t = 0; t < T; ++t) out[t] /= n;
    return out;
}

// simple normal CDF helper
inline double normal_cdf(double z) {
    return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
}

// Empirical Bayes learner (simplified, per-run)
void empirical_bayes(int T, int n, int arms,
                    const Mat &theta, double sigma_r,
                    Vec mu0, Vec tau0,
                    IMat &actions, Mat &rewards,
                    std::mt19937_64 &rng) {
    actions.assign(n, IVec(T));
    rewards.assign(n, Vec(T));

    // per-person posterior means/vars
    Mat m(n, Vec(arms));
    Mat v(n, Vec(arms));
    for (int i = 0; i < n; ++i) for (int a = 0; a < arms; ++a) {
        m[i][a] = mu0[a];
        v[i][a] = tau0[a]*tau0[a];
    }

    IVec N(n*arms, 0); // not used heavily but keep for parity

    double sig2 = sigma_r*sigma_r;

    for (int t = 0; t < T; ++t) {
        // estimate hyperparams per arm (method-of-moments)
        Vec mu_hat(arms, 0.0), tau2_hat(arms, 0.0);
        for (int a = 0; a < arms; ++a) {
            // mean of m[:,a]
            double mean_m = 0.0;
            for (int i = 0; i < n; ++i) mean_m += m[i][a];
            mean_m /= n;
            // variance of m[:,a]
            double var_m = 0.0;
            for (int i = 0; i < n; ++i) {
                double d = m[i][a] - mean_m;
                var_m += d*d;
            }
            var_m = (n > 1) ? var_m / (n - 1) : 0.0;
            // mean of v
            double mean_v = 0.0;
            for (int i = 0; i < n; ++i) mean_v += v[i][a];
            mean_v /= n;
            double est_tau2 = var_m - mean_v;
            if (est_tau2 < 0.0) est_tau2 = 0.0;
            mu_hat[a] = mean_m;
            tau2_hat[a] = est_tau2;
        }

        // EB shrinkage and selection per participant
        for (int i = 0; i < n; ++i) {
            Vec m_eb(arms);
            Vec v_eb(arms);
            for (int a = 0; a < arms; ++a) {
                double lam = (tau2_hat[a] > 0.0) ? (tau2_hat[a] / (tau2_hat[a] + v[i][a])) : 1.0;
                m_eb[a] = lam * m[i][a] + (1.0 - lam) * mu_hat[a];
                if (tau2_hat[a] > 0.0) v_eb[a] = (v[i][a] * tau2_hat[a]) / (v[i][a] + tau2_hat[a]);
                else v_eb[a] = 0.0;
            }

            int a_sel = 0;
            if (arms == 2) {
                double diff = m_eb[1] - m_eb[0];
                double var_diff = v_eb[1] + v_eb[0];
                double z = diff / std::sqrt(std::max(var_diff, 1e-20));
                double p1 = normal_cdf(z);
                std::uniform_real_distribution<double> ud(0.0, 1.0);
                a_sel = (ud(rng) < p1) ? 1 : 0;
            } else {
                double best = -INFINITY;
                for (int a = 0; a < arms; ++a) {
                    std::normal_distribution<double> dist(m_eb[a], std::sqrt(std::max(v_eb[a], 0.0)));
                    double s = dist(rng);
                    if (s > best) { best = s; a_sel = a; }
                }
            }

            std::normal_distribution<double> rd(theta[i][a_sel], sigma_r);
            double r = rd(rng);
            actions[i][t] = a_sel;
            rewards[i][t] = r;

            // posterior update for chosen arm (Gaussian conjugate)
            double sig2_choice = sigma_r*sigma_r;
            double prec = 1.0 / v[i][a_sel] + 1.0 / sig2_choice;
            double v_new = 1.0 / prec;
            double m_new = v_new * (m[i][a_sel] / v[i][a_sel] + r / sig2_choice);
            m[i][a_sel] = m_new;
            v[i][a_sel] = v_new;
        }
    }
}

int main(){
    // Scenario parameters (you can change these)
    std::vector<int> ns = {50};
    std::vector<double> mus = {0.0, 3.0};
    std::vector<double> taus = {0.5, 1.0, 5.0};
    int T = 200;
    Vec sigma_a = {5.0, 5.0};
    int runs = 100; // match the user's example; increase if desired

    // priors
    Vec mu0 = {0.0, 0.0};
    Vec tau0 = {0.1, 0.1};

    std::ofstream out("regret_scenarios.csv");
    out << "n,mu2,tau,t,mean_unpooled,se_unpooled,mean_pooled,se_pooled,mean_eb,se_eb\n";

    for (int n : ns) {
        for (double mu2 : mus) {
            for (double tau : taus) {
                // accumulators: sum and sumsq per algorithm per time
                Vec sum_unp(T, 0.0), sumsq_unp(T, 0.0);
                Vec sum_pool(T, 0.0), sumsq_pool(T, 0.0);
                Vec sum_eb(T, 0.0), sumsq_eb(T, 0.0);

                for (int r = 0; r < runs; ++r) {
                    std::mt19937_64 rng(r + 1000);
                    Vec mu_a = {0.0, mu2};
                    Vec tau_a = {tau, tau};
                    Mat theta = set_environment(n, 2, mu_a, tau_a, rng);

                    IMat acts_unp, acts_pool, acts_eb;
                    Mat rews_unp, rews_pool, rews_eb;

                    thompson_sampling(T, n, 2, theta, sigma_a[0], mu0, tau0, acts_unp, rews_unp, rng);
                    pooled_thompson_sampling(T, n, 2, theta, sigma_a[0], mu0, tau0, acts_pool, rews_pool, rng);
                    empirical_bayes(T, n, 2, theta, sigma_a[0], mu0, tau0, acts_eb, rews_eb, rng);

                    Vec reg_unp = cumulative_regret(theta, acts_unp);
                    Vec reg_pool = cumulative_regret(theta, acts_pool);
                    Vec reg_ebb = cumulative_regret(theta, acts_eb);

                    for (int t = 0; t < T; ++t) {
                        sum_unp[t] += reg_unp[t];
                        sumsq_unp[t] += reg_unp[t]*reg_unp[t];
                        sum_pool[t] += reg_pool[t];
                        sumsq_pool[t] += reg_pool[t]*reg_pool[t];
                        sum_eb[t] += reg_ebb[t];
                        sumsq_eb[t] += reg_ebb[t]*reg_ebb[t];
                    }
                }

                // write means and standard errors
                for (int t = 0; t < T; ++t) {
                    double mean_unp = sum_unp[t] / runs;
                    double mean_pool = sum_pool[t] / runs;
                    double mean_ebv = sum_eb[t] / runs;

                    double var_unp = (runs>1) ? (sumsq_unp[t] - runs * mean_unp * mean_unp) / (runs - 1) : 0.0;
                    double var_pool = (runs>1) ? (sumsq_pool[t] - runs * mean_pool * mean_pool) / (runs - 1) : 0.0;
                    double var_ebv = (runs>1) ? (sumsq_eb[t] - runs * mean_ebv * mean_ebv) / (runs - 1) : 0.0;

                    double se_unp = std::sqrt(std::max(0.0, var_unp)) / std::sqrt(runs);
                    double se_pool = std::sqrt(std::max(0.0, var_pool)) / std::sqrt(runs);
                    double se_eb = std::sqrt(std::max(0.0, var_ebv)) / std::sqrt(runs);

                    out << n << ',' << mu2 << ',' << tau << ',' << t << ','
                        << std::setprecision(8) << mean_unp << ',' << se_unp << ','
                        << mean_pool << ',' << se_pool << ','
                        << mean_ebv << ',' << se_eb << '\n';
                }
                std::cout << "Wrote scenario n="<<n<<" mu2="<<mu2<<" tau="<<tau<<"\n";
            }
        }
    }

    out.close();
    std::cout << "Wrote regret_scenarios.csv (" << runs << " runs per scenario)\n";
    return 0;
}
