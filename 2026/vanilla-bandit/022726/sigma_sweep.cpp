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

// create environment: theta ~ Normal(mu_a, tau_a)
Mat set_environment(int n, int arms, const Vec &mu_a, const Vec &tau_a, std::mt19937_64 &rng) {
    Mat theta(n, Vec(arms));
    for (int a=0; a<arms; ++a) {
        std::normal_distribution<double> d(mu_a[a], tau_a[a]);
        for (int i=0; i<n; ++i) theta[i][a] = d(rng);
    }
    return theta;
}

// unpooled Thompson sampling - returns final cumulative regret
double thompson_sampling(int T, int n, const Mat &theta, double sigma_r,
                        Vec mu0, Vec tau0, std::mt19937_64 &rng) {
    IMat N(n, IVec(2,0));
    Mat S(n, Vec(2,0.0));
    double sig2 = sigma_r*sigma_r;
    double total_regret = 0.0;
    
    for (int t=0; t<T; ++t) {
        for (int i=0; i<n; ++i) {
            // posterior for both arms
            Vec post_mean(2), post_var(2);
            for (int a=0; a<2; ++a) {
                double prec_prior = 1.0/(tau0[a]*tau0[a]);
                double prec_data  = N[i][a] / sig2;
                double v = 1.0/(prec_data + prec_prior);
                double m = v*(mu0[a]/(tau0[a]*tau0[a]) + S[i][a]/sig2);
                if (N[i][a]==0) { m = mu0[a]; v = tau0[a]*tau0[a]; }
                post_mean[a] = m;
                post_var[a]  = v;
            }
            
            // sample and choose
            double best_val=-INFINITY;
            int best_a=0;
            for (int a=0; a<2; ++a) {
                std::normal_distribution<double> dist(post_mean[a], std::sqrt(post_var[a]));
                double sample = dist(rng);
                if (sample>best_val) { best_val=sample; best_a=a; }
            }
            
            // draw reward
            std::normal_distribution<double> rd(theta[i][best_a], sigma_r);
            double r = rd(rng);
            N[i][best_a]++;
            S[i][best_a] += r;
            
            // accumulate regret
            double best_theta = std::max(theta[i][0], theta[i][1]);
            total_regret += (best_theta - theta[i][best_a]);
        }
    }
    return total_regret / n; // average per person
}

// pooled Thompson sampling
double pooled_thompson_sampling(int T, int n, const Mat &theta, double sigma_r,
                               Vec mu0, Vec tau0, std::mt19937_64 &rng) {
    IVec N(2,0);
    Vec S(2,0.0);
    double sig2 = sigma_r*sigma_r;
    double total_regret = 0.0;
    
    for(int t=0; t<T; ++t){
        Vec post_mean(2), post_var(2);
        for(int a=0; a<2; ++a){
            double prec_prior = 1.0/(tau0[a]*tau0[a]);
            double prec_data  = N[a]/sig2;
            double v = 1.0/(prec_data+prec_prior);
            double m = v*(mu0[a]/(tau0[a]*tau0[a]) + S[a]/sig2);
            if(N[a]==0){m=mu0[a]; v=tau0[a]*tau0[a];}
            post_mean[a]=m; post_var[a]=v;
        }

        // each participant independently samples from the shared posterior
        for (int i = 0; i < n; ++i) {
            double best_val = -INFINITY;
            int best_a = 0;
            for (int a = 0; a < 2; ++a) {
                std::normal_distribution<double> dist(post_mean[a], std::sqrt(post_var[a]));
                double samp = dist(rng);
                if (samp > best_val) { best_val = samp; best_a = a; }
            }

            std::normal_distribution<double> rd(theta[i][best_a], sigma_r);
            double r = rd(rng);
            S[best_a] += r;
            N[best_a]++;

            double best_theta = std::max(theta[i][0], theta[i][1]);
            total_regret += (best_theta - theta[i][best_a]);
        }
        
        // int best_a=0;
        // double best_val=-INFINITY;
        // for(int a=0; a<2; ++a){
        //     std::normal_distribution<double> dist(post_mean[a], std::sqrt(post_var[a]));
        //     double samp=dist(rng);
        //     if(samp>best_val){best_val=samp; best_a=a;}
        // }
        
        // for(int i=0; i<n; ++i){
        //     std::normal_distribution<double> rd(theta[i][best_a], sigma_r);
        //     double r=rd(rng);
        //     S[best_a] += r;
            
        //     double best_theta = std::max(theta[i][0], theta[i][1]);
        //     total_regret += (best_theta - theta[i][best_a]);
        // }
        // N[best_a]+=n;
    }
    return total_regret / n;
}

inline double normal_cdf(double z) {
    return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
}

// Empirical Bayes learner
double empirical_bayes(int T, int n, const Mat &theta, double sigma_r,
                      Vec mu0, Vec tau0, std::mt19937_64 &rng) {
    Mat m(n, Vec(2));
    Mat v(n, Vec(2));
    for (int i = 0; i < n; ++i) {
        m[i][0] = mu0[0]; m[i][1] = mu0[1];
        v[i][0] = tau0[0]*tau0[0]; v[i][1] = tau0[1]*tau0[1];
    }
    
    double sig2 = sigma_r*sigma_r;
    double total_regret = 0.0;
    
    for (int t = 0; t < T; ++t) {
        // estimate hyperparams per arm
        Vec mu_hat(2, 0.0), tau2_hat(2, 0.0);
        for (int a = 0; a < 2; ++a) {
            double mean_m = 0.0;
            for (int i = 0; i < n; ++i) mean_m += m[i][a];
            mean_m /= n;
            
            double var_m = 0.0;
            for (int i = 0; i < n; ++i) {
                double d = m[i][a] - mean_m;
                var_m += d*d;
            }
            var_m = (n > 1) ? var_m / (n - 1) : 0.0;
            
            double mean_v = 0.0;
            for (int i = 0; i < n; ++i) mean_v += v[i][a];
            mean_v /= n;
            
            double est_tau2 = var_m - mean_v;
            if (est_tau2 < 0.0) est_tau2 = 0.0;
            // mu_hat[a] = mean_m;

            double wsum = 0.0, wmsum = 0.0;
            for (int i = 0; i < n; ++i) {
                double w = 1.0 / (tau2_hat[a] + v[i][a]);
                wsum  += w;
                wmsum += w * m[i][a];
            }
            mu_hat[a] = (wsum > 0.0) ? wmsum / wsum : mean_m;

            tau2_hat[a] = est_tau2;
        }
        
        // EB shrinkage and selection per participant
        for (int i = 0; i < n; ++i) {
            Vec m_eb(2);
            Vec v_eb(2);
            for (int a = 0; a < 2; ++a) {
                // double lam = (tau2_hat[a] > 0.0) ? (tau2_hat[a] / (tau2_hat[a] + v[i][a])) : 1.0;
                double lam = (tau2_hat[a] > 0.0) ? (tau2_hat[a] / (tau2_hat[a] + v[i][a])) : 0.0;

                m_eb[a] = lam * m[i][a] + (1.0 - lam) * mu_hat[a];
                if (tau2_hat[a] > 0.0) v_eb[a] = (v[i][a] * tau2_hat[a]) / (v[i][a] + tau2_hat[a]);
                else v_eb[a] = 0.0;
            }
            
            double diff = m_eb[1] - m_eb[0];
            double var_diff = v_eb[1] + v_eb[0];
            double z = diff / std::sqrt(std::max(var_diff, 1e-20));
            double p1 = normal_cdf(z);
            std::uniform_real_distribution<double> ud(0.0, 1.0);
            int a_sel = (ud(rng) < p1) ? 1 : 0;
            
            std::normal_distribution<double> rd(theta[i][a_sel], sigma_r);
            double r = rd(rng);
            
            // posterior update
            double prec = 1.0 / v[i][a_sel] + 1.0 / sig2;
            double v_new = 1.0 / prec;
            double m_new = v_new * (m[i][a_sel] / v[i][a_sel] + r / sig2);
            m[i][a_sel] = m_new;
            v[i][a_sel] = v_new;
            
            // accumulate regret
            double best_theta = std::max(theta[i][0], theta[i][1]);
            total_regret += (best_theta - theta[i][a_sel]);
        }
    }
    return total_regret / n;
}

int main(){
    // Fixed parameters
    int n = 50;
    int T = 100;
    double mu2 = 1.0;  // true difference
    double tau = 1.0;  // between-person heterogeneity (fixed)
    int runs = 500;
    
    Vec mu0 = {0.0, 0.0};
    Vec tau0 = {1.0, 5.0};
    
    // Build custom sigma values: 0.01-0.1 by 0.01, 0.1-1 by 0.1, 1-10 by 0.5
    Vec sigma_values;
    
    // 0.01 to 0.1 by 0.01
    for (double sigma = 0.01; sigma <= 0.1 + 1e-9; sigma += 0.01) {
        sigma_values.push_back(sigma);
    }
    
    // 0.2 to 1.0 by 0.1 (0.1 already added)
    for (double sigma = 0.2; sigma <= 1.0 + 1e-9; sigma += 0.1) {
        sigma_values.push_back(sigma);
    }
    
    // 1.5 to 10.0 by 0.5 (1.0 already added)
    for (double sigma = 1.5; sigma <= 10.0 + 1e-9; sigma += 0.5) {
        sigma_values.push_back(sigma);
    }
    
    int n_sigma = sigma_values.size();
    
    std::ofstream out("final_graphs_midterm/different_taus_final.csv");
    out << "sigma,mean_unpooled,se_unpooled,mean_pooled,se_pooled,mean_eb,se_eb,winner\n";
    
    std::cout << "Sweeping " << n_sigma << " sigma values from " << sigma_values[0] 
              << " to " << sigma_values[n_sigma-1] << "...\n";
    
    for (int sigma_idx = 0; sigma_idx < n_sigma; ++sigma_idx) {
        double sigma = sigma_values[sigma_idx];
        
        double sum_unp = 0.0, sum_pool = 0.0, sum_eb = 0.0;
        double sumsq_unp = 0.0, sumsq_pool = 0.0, sumsq_eb = 0.0;
        
        // Run simulations
        for (int r = 0; r < runs; ++r) {
            std::mt19937_64 rng(r + 10000 + sigma_idx * 1000);
            Vec mu_a = {0.0, mu2};
            Vec tau_a = {tau, tau};
            Mat theta = set_environment(n, 2, mu_a, tau_a, rng);
            
            double reg_unp = thompson_sampling(T, n, theta, sigma, mu0, tau0, rng);
            double reg_pool = pooled_thompson_sampling(T, n, theta, sigma, mu0, tau0, rng);
            double reg_eb = empirical_bayes(T, n, theta, sigma, mu0, tau0, rng);
            
            sum_unp += reg_unp;
            sum_pool += reg_pool;
            sum_eb += reg_eb;
            
            sumsq_unp += reg_unp * reg_unp;
            sumsq_pool += reg_pool * reg_pool;
            sumsq_eb += reg_eb * reg_eb;
        }
        
        double mean_unp = sum_unp / runs;
        double mean_pool = sum_pool / runs;
        double mean_eb = sum_eb / runs;
        
        // Calculate standard errors
        double var_unp = (sumsq_unp - runs * mean_unp * mean_unp) / (runs - 1);
        double var_pool = (sumsq_pool - runs * mean_pool * mean_pool) / (runs - 1);
        double var_eb = (sumsq_eb - runs * mean_eb * mean_eb) / (runs - 1);
        
        double se_unp = std::sqrt(std::max(0.0, var_unp)) / std::sqrt(runs);
        double se_pool = std::sqrt(std::max(0.0, var_pool)) / std::sqrt(runs);
        double se_eb = std::sqrt(std::max(0.0, var_eb)) / std::sqrt(runs);
        
        // Determine winner
        std::string winner;
        double min_regret = std::min({mean_unp, mean_pool, mean_eb});
        if (mean_unp == min_regret) winner = "Unpooled";
        else if (mean_pool == min_regret) winner = "Pooled";
        else winner = "EB";
        
        out << std::fixed << std::setprecision(4) 
            << sigma << ',' << mean_unp << ',' << se_unp << ',' 
            << mean_pool << ',' << se_pool << ',' 
            << mean_eb << ',' << se_eb << ',' << winner << '\n';
        
        if ((sigma_idx + 1) % 5 == 0) {
            std::cout << "Progress: " << (sigma_idx + 1) << "/" << n_sigma 
                     << " (sigma=" << std::fixed << std::setprecision(2) << sigma << ")\n";
        }
    }
    
    out.close();
    std::cout << "\nDone! Results written to final_graphs_midterm/different_taus_final.csv\n";
    std::cout << "Summary: Check which algorithm wins in different noise regimes.\n";
    
    return 0;
}
