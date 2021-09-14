set lmargin 9
set grid lc rgb '#f0f0f0'
set xlabel 'MD time unit ($\delta$)'
set ylabel '$\Delta Q^2 = \langle (Q_{\tau+\delta}-Q_\tau)^2 \rangle$'
set key left Left reverse
set xrange [0:70]
pf(f)="<sed -n '/^beta: 5$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/".f."_dq2_post_train"
plot\
	pf("t_hmc_2") u ($1*20*0.13025032951094404*0.636829):2:4 w e t 'HMC, 20 steps/traj',\
	pf("t_hmc_0") u ($1*40*0.083048177247344687*0.851581):2:4 w e t 'HMC, 40 steps/traj',\
	pf("t_force_b25_0") u ($1*10*0.1*0.914215):2:4 w e t 'TF-2.5.1 CPU, FTHMC trained at $\beta=5$, 10 steps/traj, $N_b=64$, $l=0.001$',\
	pf("t_force_b25_0.1") u ($1*10*0.1*0.917694):2:4 w e t 'TF-2.6.0 CPU, FTHMC trained at $\beta=5$, 10 steps/traj, $N_b=64$, $l=0.001$',\
	pf("t_force_b25_0.mac") u ($1*10*0.1*0.919098):2:4 w e t 'TF-2.6.0 CPU macOS, FTHMC trained at $\beta=5$, 10 steps/traj, $N_b=64$, $l=0.001$',\
	pf("t_force_b25_0.cuda") u ($1*10*0.1*0.918655):2:4 w e t 'TF-2.6.0 GPU, FTHMC trained at $\beta=5$, 10 steps/traj, $N_b=64$, $l=0.001$',\
	pf("t_force_b25_0s.cuda") u ($1*10*0.1*0.0571899):2:4 w e t 'TF-2.6.0 CPU, FTHMC trained at $\beta=5$, 10 steps/traj, $N_b=64$, $l=0.001$',\
	pf("t_force_b25_0s0") u ($1*10*0.1*0.937996):2:4 w e t 'TF-2.6.0 CPU, FTHMC trained at $\beta=5$, 10 steps/traj, $N_b=128$, $l=0.0005$',\
	pf("t_force_b25_0s1") u ($1*10*0.1*0.937752):2:4 w e t 'TF-2.6.0 CPU, FTHMC trained at $\beta=5$, 10 steps/traj, $N_b=128$, $l=0.00025$',\
	pf("t_force_b25_1") u ($1*10*0.1*0.934906):2:4 w e t 'TF-2.5.1 CPU, FTHMC trained at $\beta=5$, 10 steps/traj, $N_b=64$, $l=0.0001$',\
	pf("t_force_b25_1.1") u ($1*10*0.1*0.928802):2:4 w e t 'TF-2.6.0 CPU, FTHMC trained at $\beta=5$, 10 steps/traj, $N_b=64$, $l=0.0001$',\
	pf("t_force_b25_1.cuda") u ($1*10*0.1*0.930725):2:4 w e t 'TF-2.6.0 GPU, FTHMC trained at $\beta=5$, 10 steps/traj, $N_b=64$, $l=0.0001$',\
	pf("t_force_b25_1l.cuda") u ($1*10*0.1*0.931107):2:4 w e t 'TF-2.6.0 GPU, FTHMC trained at $\beta=5$, 10 steps/traj, $N_b=64$, $l=0.00025$'
