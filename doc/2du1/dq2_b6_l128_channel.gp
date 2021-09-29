set lmargin 9
set grid lc rgb '#f0f0f0'
set xlabel 'MD time unit ($\delta$) $×$ acceptance rate'
set ylabel '$\Delta Q^2/V\times 1024 = \langle (Q_{\tau+\delta}-Q_\tau)^2 \rangle/V\times 1024$'
set key left Left reverse
pf(f)="<sed -n '/^beta: 6$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/".f."_dq2_post_train"
set xrange [0:8]
plot\
	pf("t_hmc_5") u ($1*20*0.02977805536756897*0.875481):($2/16.):($4/16.) w e t 'HMC, 20 steps/traj',\
	pf("t_force_b25_0s6a") u ($1*10*0.0125*0.98584):($2/16.):($4/16.) w e t 'FTHMC, 10 steps/traj, trained batch size 2',\
	pf("t_force_b25_0s6ac") u ($1*10*0.0125*0.989258):($2/16.):($4/16.) w e t 'FTHMC, 10 steps/traj, trained batch size 2, 3 channel ConvNets',\
	pf("t_force_b25_0s6a_t2") u ($1*10*0.0125*0.98877):($2/16.):($4/16.) w e t 'FTHMC, 10 steps/traj, trained batch size 2, double transforms'
