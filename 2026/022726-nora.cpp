#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>

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

int main(){
    int n=50, arms=2, T=200;
    Vec mu_a={0.0,0.0}, tau_a={0.01,0.01};
    double sigma_r=0.5;
    Vec mu0={0.0,0.0}, tau0={0.1,0.1};

    int runs = 1000;
    Vec avg_ts(T,0.0), avg_pts(T,0.0);

    for (int r = 0; r < runs; ++r) {
        std::mt19937_64 rng(r);
        Mat theta = set_environment(n,arms,mu_a,tau_a,rng);
        IMat acts_ts, acts_pts;
        Mat rews_ts, rews_pts;

        thompson_sampling(T,n,arms,theta,sigma_r,mu0,tau0,acts_ts,rews_ts,rng);
        pooled_thompson_sampling(T,n,arms,theta,sigma_r,mu0,tau0,acts_pts,rews_pts,rng);

        Vec reg_ts  = cumulative_regret(theta, acts_ts);
        Vec reg_pts = cumulative_regret(theta, acts_pts);
        for (int t = 0; t < T; ++t) {
            avg_ts[t] += reg_ts[t];
            avg_pts[t] += reg_pts[t];
        }
    }
    for (int t = 0; t < T; ++t) {
        avg_ts[t] /= runs;
        avg_pts[t] /= runs;
    }

    std::ofstream out("regret.csv");
    out << "t,unpooled,pooled\n";
    for (int t = 0; t < T; ++t) {
        out << t << ',' << avg_ts[t] << ',' << avg_pts[t] << '\n';
    }
    out.close();

    std::cout << "Wrote regret.csv (" << runs << " runs)\n";
    return 0;
}
