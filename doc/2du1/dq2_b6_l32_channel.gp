set lmargin 9
set grid lc rgb '#f0f0f0'
set xlabel 'MD time unit ($\delta$) $×$ acceptance rate'
set ylabel '$\Delta Q^2 = \langle (Q_{\tau+\delta}-Q_\tau)^2 \rangle$'
set key left Left reverse
pf(f)="<sed -n '/^beta: 6$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/".f."_dq2_post_train"
set xrange [0:30]
plot\
	pf("t_hmc_3") u ($1*20*0.033155120545061763*0.949135):2:4 w e t 'HMC, 20 steps/traj',\
	pf("t_hmc_1") u ($1*40*0.07769028569149361*0.672693):2:4 w e t 'HMC, 40 steps/traj',\
	pf("t_force_b25_0s2a") u ($1*10*0.05*0.928131):2:4 w e t 'FTHMC, 10 steps/traj, trained batch size 32, different init',\
	pf("t_force_b25_0s2ac") u ($1*10*0.05*0.927032):2:4 w e t 'FTHMC, 10 steps/traj, trained batch size 32, 3 channel ConvNets'
