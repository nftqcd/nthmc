set lmargin 9
set grid lc rgb '#f0f0f0'
set xlabel 'MD time unit ($\delta$) $×$ acceptance rate'
set ylabel '$\Delta Q^2/V\times 1024 = \langle (Q_{\tau+\delta}-Q_\tau)^2 \rangle/V\times 1024$'
set key left Left reverse
pf(f)="<sed -n '/^beta: 6$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/".f."_dq2_post_train"
set xrange [0:60]
plot\
	pf("t_hmc_2") u ($1*20*0.074845677855114881*0.86284):($2*4.):($4*4.) w e t 'HMC, 20 steps/traj',\
	pf("t_force_b25_0s0a") u ($1*10*0.1*0.90332):($2*4.):($4*4.) w e t 'FTHMC, 10 steps/traj, trained batch size 128',\
	pf("t_force_b25_0s0ac") u ($1*10*0.1*0.895744):($2*4.):($4*4.) w e t 'FTHMC, 10 steps/traj, trained batch size 128, 3 channel ConvNets',\
	pf("t_force_b25_0s0a_t2") u ($1*10*0.1*0.832382):($2*4.):($4*4.) w e t 'FTHMC, 10 steps/traj, trained batch size 128, double transforms'
