set lmargin 9
set grid lc rgb '#f0f0f0'
set xlabel 'MD time unit ($\delta$) $×$ acceptance rate'
set ylabel '$\Delta Q^2 = \langle (Q_{\tau+\delta}-Q_\tau)^2 \rangle$'
set key left Left reverse
pf(f)="<sed -n '/^beta: 6$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/".f."_dq2_post_train"
set xrange [0:15]
plot\
	pf("t_hmc_4") u ($1*20*0.033498063973276447*0.894562):2:4 w e t 'HMC, 20 steps/traj',\
	pf("t_force_b25_0s4a") u ($1*10*0.025*0.967407):2:4 w e t 'FTHMC, 10 steps/traj, trained batch size 8, different init',\
	pf("t_force_b25_0s4ac") u ($1*10*0.025*0.961426):2:4 w e t 'FTHMC, 10 steps/traj, trained batch size 8, 3 channel ConvNets'
