set grid lc rgb '#f0f0f0'

set tmargin 6.5
set bmargin 4
set lmargin 10
set rmargin 0

set log y
set yrange [0.134:0.213]
set multiplot

set origin 0.0,0
set size 0.34,1

set key tmargin left Left reverse
set ytics ('$1/5$' 1./5, '$1/6$' 1./6, '$1/7$' 1./7)
set ylabel '$1/\beta$'
set xlabel 'acceptance'
set xrange [0.68:0.92]
set xtics 0.1
plot\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_hmc_l64_params_accept_post_train   |paste - - - -" u 8:(1./$2):10 w xerror t 'HMC',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b5_accept_post_train     |paste - - - -" u 8:(1./$2):10 w xerror t 'NTHMC trained at $\beta=5$',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b6_accept_post_train     |paste - - - -|head -n1;grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b6_startb6_accept_post_train     |paste - - - -|head -n1;grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b6_startb7_accept_post_train     |paste - - - -" u 8:(1./$2):10 w xerror t 'NTHMC trained at $\beta=6$',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b6_5cn8_5cn10_lr1e-5_accept_post_train |paste - - - -" u 8:(1./$2):10 w xerror t 'NTHMC$^\dag$ trained at $\beta=6$'

set origin 0.34,0
set size 0.32,1

set lmargin 0

unset key
set ytics ('' 1./5, '' 1./6, '' 1./7)
unset ylabel
set xlabel 'step size'
set xrange [-0.03:0.28]
set xtics 0.1
plot\
	"<paste - - <../../u1_2d/stats/s_hmc_l64_params_stepsize_train   |awk '$2>=5&&$2<=7'" u 4:(1./$2) w lp t '',\
	"<paste - - <../../u1_2d/stats/s_nthmc_l64_b5_stepsize_train     |awk '$2>=5&&$2<=7'" u 4:(1./$2) w lp t '',\
	"<paste - - <../../u1_2d/stats/s_nthmc_l64_b6_stepsize_train     |awk '$2>=5&&$2<=5'; paste - - <../../u1_2d/stats/s_nthmc_l64_b6_startb6_stepsize_train     |awk '$2>=6&&$2<=6'; paste - - <../../u1_2d/stats/s_nthmc_l64_b6_startb7_stepsize_train     |awk '$2>=7&&$2<=7'" u 4:(1./$2) w lp t '',\
	"<paste - - <../../u1_2d/stats//s_nthmc_l64_b6_5cn8_5cn10_lr1e-5_stepsize_train |awk '$2>=5&&$2<=7'" u 4:(1./$2) w lp t ''

set origin 0.66,0
set size 0.34,1

set rmargin 4

set xlabel '$\gamma_{\text{HMC}}(\delta=16)\Big/\gamma_{\text{NTHMC}}(\delta=16)$
set xrange [1.7:5.3]
set xtics 1
plot\
	"<for b in 5 6 7;do ./get_dq2_hmc_scaled_b_v_d $b 64 4; ./get_dq2_hmc_scaled_b_v_d $b 64 4 s_nthmc_l64_b5;done|paste - -" u ($5/$2):(1./$1):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w xerror lt 2 t '',\
	"<(./get_dq2_hmc_scaled_b_v_d 5 64 4; ./get_dq2_hmc_scaled_b_v_d 5 64 4 s_nthmc_l64_b6;./get_dq2_hmc_scaled_b_v_d 6 64 4; ./get_dq2_hmc_scaled_b_v_d 6 64 4 s_nthmc_l64_b6_startb6;./get_dq2_hmc_scaled_b_v_d 7 64 4; ./get_dq2_hmc_scaled_b_v_d 7 64 4 s_nthmc_l64_b6_startb7)|paste - -" u ($5/$2):(1./$1):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w xerror lt 3 t '',\
	"<for b in 5 6 7;do ./get_dq2_hmc_scaled_b_v_d $b 64 4; ./get_dq2_hmc_scaled_b_v_d $b 64 4 s_nthmc_l64_b6_5cn8_5cn10_lr1e-5;done|paste - -" u ($5/$2):(1./$1):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w xerror lt 4 t ''
unset multiplot
