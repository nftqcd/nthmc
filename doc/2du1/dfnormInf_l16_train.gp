set lmargin 9
set grid lc rgb '#f0f0f0'
set xlabel 'training step'
set ylabel '$(F_{\text{tr}}-F_{\beta=2.5})^2/V$'
set key left Left reverse
pf(f)="<9 awk '/# training inference step/{p=1} p==1&&/^dfnormInf/{print;p=0}' ../../u1_2d/".f.".log"
set xrange [0:5200]
plot\
	pf("t_force_b25_0") u 0:2 w l t 'TF-2.5.1 CPU, $N_b=64$, $l=0.001$',\
	pf("t_force_b25_0.1") u 0:2 w l t 'TF-2.6.0 CPU, $N_b=64$, $l=0.001$',\
	pf("t_force_b25_0.cuda") u 0:2 w l t 'TF-2.6.0 GPU, $N_b=64$, $l=0.001$',\
	pf("t_force_b25_0s.cuda") u 0:2 w l t 'TF-2.6.0 CPU, $N_b=64$, $l=0.001$',\
	pf("t_force_b25_0s0") u 0:2 w l t 'TF-2.6.0 CPU, $N_b=128$, $l=0.0005$',\
	pf("t_force_b25_0s1") u 0:2 w l t 'TF-2.6.0 CPU, $N_b=128$, $l=0.00025$',\
	pf("t_force_b25_1") u 0:2 w l t 'TF-2.5.1 CPU, $N_b=64$, $l=0.0001$',\
	pf("t_force_b25_1.cuda") u 0:2 w l t 'TF-2.6.0 GPU, $N_b=64$, $l=0.0001$',\
	pf("t_force_b25_1l.cuda") u 0:2 w l t 'TF-2.6.0 GPU, $N_b=64$, $l=0.00025$'
