set lmargin 9
set grid lc rgb '#f0f0f0'
set xlabel 'MD time unit ($\delta$)'
set ylabel '$\Delta Q^2 = \langle (Q_{\tau+\delta}-Q_\tau)^2 \rangle$'
set key left Left reverse
set xrange [0:70]
plot\
	"<sed -n '/^beta: 5$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/t_hmc_2_dq2_post_train" u ($1*20*0.13025032951094404):2:4 w e t 'HMC, 20 steps/traj',\
	"<sed -n '/^beta: 5$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/t_hmc_0_dq2_post_train" u ($1*40*0.083048177247344687):2:4 w e t 'HMC, 40 steps/traj',\
	"<sed -n '/^beta: 5$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/t_force_b25_0_dq2_post_train" u ($1*10*0.1):2:4 w e t 'TF-2.5.1 CPU, FTHMC trained at $\beta=5$, 10 steps/traj',\
	"<sed -n '/^beta: 5$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/t_force_b25_0.1_dq2_post_train" u ($1*10*0.1):2:4 w e t 'TF-2.6.0 CPU, FTHMC trained at $\beta=5$, 10 steps/traj',\
	"<sed -n '/^beta: 5$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/t_force_b25_0.mac_dq2_post_train" u ($1*10*0.1):2:4 w e t 'TF-2.6.0 CPU macOS, FTHMC trained at $\beta=5$, 10 steps/traj',\
	"<sed -n '/^beta: 5$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/t_force_b25_0.cuda_dq2_post_train" u ($1*10*0.1):2:4 w e t 'TF-2.6.0 GPU, FTHMC trained at $\beta=5$, 10 steps/traj'
