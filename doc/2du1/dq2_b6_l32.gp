set lmargin 9
set grid lc rgb '#f0f0f0'
set xlabel 'MD time unit ($\delta$)'
set ylabel '$\Delta Q^2 = \langle (Q_{\tau+\delta}-Q_\tau)^2 \rangle$'
set key bottom
plot\
	"<sed -n '/^beta: 6$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/t_hmc_3_dq2_post_train" u ($1*20*0.033155120545061763):2:4 w e t 'HMC, 20 steps/traj',\
	"<sed -n '/^beta: 6$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/t_hmc_1_dq2_post_train" u ($1*40*0.07769028569149361):2:4 w e t 'HMC, 40 steps/traj',\
	"<sed -n '/^beta: 6$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/i_force_b25_0_b5_2_dq2_post_train" u ($1*10*0.091246348693292734):2:4 w e t 'FTHMC trained in $16^2$ at $\beta=5$, 10 steps/traj',\
	"<sed -n '/^beta: 6$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/i_force_b25_0_b5_3_dq2_post_train" u ($1*20*0.04991851344096123):2:4 w e t 'FTHMC trained in $16^2$ at $\beta=5$, 20 steps/traj'
