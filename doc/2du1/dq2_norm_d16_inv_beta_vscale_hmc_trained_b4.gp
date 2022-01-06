set lmargin 12
set grid lc rgb '#f0f0f0'
set title '$V = 64^2$, trained at $\beta=4$'
set xlabel '$1/\beta$'
set ylabel 'ratios of $\Delta Q^2(\delta=16)$ w.r.t. model trained at $\text{lr}=10^{-3}$' # delta in MDTU, 4 trajectories here
set log x
set xrange [0.134:0.266]
set xtics ('$1/4$' 1./4, '$1/5$' 1./5, '$1/6$' 1./6, '$1/7$' 1./7)
plot\
	"<for b in 4 5 6 7;do echo $(./get_dq2_hmc_scaled_b_v_d $b 64 4 s_nthmc_l64_b4) $(./get_dq2_hmc_scaled_b_v_d $b 64 4 s_nthmc_l64_b4_lr1e-3_cni10);done" u (1./$1):($5/$2):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w e t '$C_{\text{norm}_\infty}=10$',\
	"<for b in 4 5 6 7;do echo $(./get_dq2_hmc_scaled_b_v_d $b 64 4 s_nthmc_l64_b4) $(./get_dq2_hmc_scaled_b_v_d $b 64 4 s_nthmc_l64_b4_lr5e-4);done"       u (1./$1):($5/$2):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w e t '$\text{lr}=5\times 10^{-4}$',\
	"<for b in 4 5 6 7;do echo $(./get_dq2_hmc_scaled_b_v_d $b 64 4 s_nthmc_l64_b4) $(./get_dq2_hmc_scaled_b_v_d $b 64 4 s_nthmc_l64_b4_lr5e-4_cni10);done" u (1./$1):($5/$2):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w e t '$\text{lr}=5\times 10^{-4}$, $C_{\text{norm}_\infty}=10$',\
	"<for b in 4 5 6 7;do echo $(./get_dq2_hmc_scaled_b_v_d $b 64 4 s_nthmc_l64_b4) $(./get_dq2_hmc_scaled_b_v_d $b 64 4 s_nthmc_l64_b4_t2_lr5e-4);done"    u (1./$1):($5/$2):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w e t '$\text{lr}=5\times 10^{-4}$, 2 steps/config',\
	"<for b in 4 5 6 7;do echo $(./get_dq2_hmc_scaled_b_v_d $b 64 4 s_nthmc_l64_b4) $(./get_dq2_hmc_scaled_b_v_d $b 64 4 s_nthmc_l64_b4_cn8);done"          u (1./$1):($5/$2):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w e t 'with $C_{\text{norm}_8}$',\
	"<for b in 4 5 6 7;do echo $(./get_dq2_hmc_scaled_b_v_d $b 64 4 s_nthmc_l64_b4) $(./get_dq2_hmc_scaled_b_v_d $b 64 4 s_nthmc_l64_b4_cn8_cn10);done"     u (1./$1):($5/$2):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w e t 'with $C_{\text{norm}_8}$ and $C_{\text{norm}_{10}}$',\
	"<for b in 4 5 6 7;do echo $(./get_dq2_hmc_scaled_b_v_d $b 64 4 s_nthmc_l64_b4) $(./get_dq2_hmc_scaled_b_v_d $b 64 4 s_nthmc_l64_b5);done"              u (1./$1):($5/$2):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w e t 'trained at $\beta=5$',\
	"<for b in 4 5 6 7;do echo $(./get_dq2_hmc_scaled_b_v_d $b 64 4 s_nthmc_l64_b4) $(./get_dq2_hmc_scaled_b_v_d $b 64 4 s_nthmc_l64_b5_cn10);done"         u (1./$1):($5/$2):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w e t 'trained at $\beta=5$ with $C_{\text{norm}_{10}}$'
