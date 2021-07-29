set lmargin 9
set grid lc rgb '#f0f0f0'
set xlabel 'MD time unit ($\delta$)'
set ylabel '$\Delta Q^2 = \langle (Q_{\tau+\delta}-Q_\tau)^2 \rangle$'
set key left Left reverse
plot\
	"<sed -n '/^beta: 6$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/t_hmc_2_dq2_post_train" u ($1*20*0.074845677855114881):2:4 w e t 'HMC, 20 steps/traj',\
	"<sed -n '/^beta: 6$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/t_hmc_0_dq2_post_train" u ($1*40*0.074065811732683923):2:4 w e t 'HMC, 40 steps/traj',\
	"<sed -n '/^beta: 6$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/t_force_b25_0_dq2_post_train" u ($1*10*0.1):2:4 w e t 'FTHMC trained at $\beta=6$, 10 steps/traj',\
	"<sed -n '/^beta: 6$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/i_force_b25_0_b5_0_dq2_post_train" u ($1*20*0.10272916993447123):2:4 w e t 'FTHMC trained at $\beta=5$, 20 steps/traj',\
	"<sed -n '/^beta: 6$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/i_force_b25_0_b5_1_dq2_inference" u ($1*40*0.1):2:4 w e t 'FTHMC trained at $\beta=5$, 40 steps/traj'
