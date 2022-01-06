set lmargin 9
set grid lc rgb '#f0f0f0'
set xlabel 'training step'
set ylabel '$\max |F_{\text{tr}}-F_{\beta=2.5}|$'
set key left Left reverse
pf(f)="<9 awk '/done mixing epoch 3/{e=1} /done training epoch 3/{e=0} /# training inference step/{p=1} e==1&&p==1&&/^dfnormInf/{print;p=0}' ../../u1_2d/".f.".log"
plot\
	pf("t_force_b25_0s2a") u 0:2 w d t 'FTHMC, 10 steps/traj, trained batch size 32',\
	pf("t_force_b25_0s2ac") u 0:2 w d t 'FTHMC, 10 steps/traj, trained batch size 32, 3 channel ConvNets',\
	pf("t_force_b25_0s2a_t2") u 0:2 w d t 'FTHMC, 10 steps/traj, trained batch size 32, double transforms'

