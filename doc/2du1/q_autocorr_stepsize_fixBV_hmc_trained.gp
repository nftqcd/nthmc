set grid lc rgb '#f0f0f0'

set tmargin 6.5
set bmargin 4
set lmargin 10
set rmargin 0

set log y
set yrange [0.104:0.213]
set multiplot

set origin 0.0,0
set size 0.34,1

set key tmargin left Left reverse
set ytics ('$1/5$' 1./5, '$1/6$' 1./6, '$1/7$' 1./7, '$1/8$' 1./8, '$1/9$' 1./9)
set ylabel '$1/\beta$'
set xlabel 'acceptance'
set xrange [0.68:0.92]
set xtics 0.1
plot\
	"<for v in 64 66 68 70 72 74 76;do ./get_accept_hmc_scaled_b_v 5 $v;done" u 8:(1./$2):10 w xerror t 'HMC $V/\beta=819.20$',\
	"<for v in 64 66 68 70 72 74 76;do ./get_accept_hmc_scaled_b_v 6 $v;done" u 8:(1./$2):10 w xerror t 'HMC $V/\beta=682.67$',\
	"<./get_accept_hmc_scaled_b_v 5 64 s_nthmc_l64_b5        ;for v in 66 68 70 72 74 76;do ./get_accept_hmc_scaled_b_v 5 $v s_nthmc_l${v}_fixBV_l64_b5;done" u 8:(1./$2):10 w xerror t 'NTHMC $V/\beta=819.20$ trained at $\beta=5$, $V=64^2$',\
	"<./get_accept_hmc_scaled_b_v 6 64 s_nthmc_l64_b6_startb6;for v in 66 68 70 72 74 76;do ./get_accept_hmc_scaled_b_v 6 $v s_nthmc_l${v}_fixBV_l64_b6;done" u 8:(1./$2):10 w xerror t 'NTHMC $V/\beta=682.67$ trained at $\beta=6$, $V=64^2$'

set origin 0.34,0
set size 0.32,1

set lmargin 0

unset key
set ytics ('' 1./5, '' 1./6, '' 1./7, '' 1./8, '' 1./9)
unset ylabel
set xlabel 'step size'
set xrange [-0.03:0.28]
set xtics 0.1
plot\
	"<for v in 64 66 68 70 72 74 76;do ./get_stepsize_hmc_scaled_b_v 5 $v;done" u 4:(1./$2) w lp t 'HMC $V/\beta=819.20$',\
	"<for v in 64 66 68 70 72 74 76;do ./get_stepsize_hmc_scaled_b_v 6 $v;done" u 4:(1./$2) w lp t 'HMC $V/\beta=682.67$',\
	"<./get_stepsize_hmc_scaled_b_v 5 64 s_nthmc_l64_b5        ;for v in 66 68 70 72 74 76;do ./get_stepsize_hmc_scaled_b_v 5 $v s_nthmc_l${v}_fixBV_l64_b5;done" u 4:(1./$2) w lp t 'NTHMC $V/\beta=819.20$ trained at $\beta=5$, $V=64^2$',\
	"<./get_stepsize_hmc_scaled_b_v 6 64 s_nthmc_l64_b6_startb6;for v in 66 68 70 72 74 76;do ./get_stepsize_hmc_scaled_b_v 6 $v s_nthmc_l${v}_fixBV_l64_b6;done" u 4:(1./$2) w lp t 'NTHMC $V/\beta=682.67$ trained at $\beta=6$, $V=64^2$'

set origin 0.66,0
set size 0.34,1

set rmargin 4

set xlabel '$\gamma_{\text{HMC}}(\delta=16)\Big/\gamma_{\text{NTHMC}}(\delta=16)$
set xrange [1.7:5.3]
set xtics 1
plot\
	"<(./get_dq2_hmc_scaled_b_v_d 5 64 4;./get_dq2_hmc_scaled_b_v_d 5 64 4 s_nthmc_l64_b5        ;for v in 66 68 70 72 74 76;do ./get_dq2_hmc_scaled_b_v_d 5 $v 4;./get_dq2_hmc_scaled_b_v_d 5 $v 4 s_nthmc_l${v}_fixBV_l64_b5;done)|paste - -" u ($5/$2):(1./$1):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w xerror lt 3 t 'NTHMC $V/\beta=819.20$ trained at $\beta=5$, $V=64^2$',\
	"<(./get_dq2_hmc_scaled_b_v_d 6 64 4;./get_dq2_hmc_scaled_b_v_d 6 64 4 s_nthmc_l64_b6_startb6;for v in 66 68 70 72 74 76;do ./get_dq2_hmc_scaled_b_v_d 6 $v 4;./get_dq2_hmc_scaled_b_v_d 6 $v 4 s_nthmc_l${v}_fixBV_l64_b6;done)|paste - -" u ($5/$2):(1./$1):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w xerror lt 4 t 'NTHMC $V/\beta=682.67$ trained at $\beta=6$, $V=64^2$'
unset multiplot
