set grid lc rgb '#a0a0a0'

set tmargin 10
set bmargin 4
set lmargin 10
set rmargin 0

set log y
set yrange [0.134:0.213]
set multiplot

figw=(1.0-0.04)/7.0    # excluding left & right margins
orig=0.0
orig_next=orig+figw+0.02    # left margin

set origin orig,0
set size orig_next-orig,1
orig=orig_next
orig_next=orig_next+figw

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
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b6_5cn8_5cn10_lr1e-5_accept_post_train |paste - - - -" u 8:(1./$2):10 w xerror t 'NTHMC$^\dag$ trained at $\beta=6$ with norm-8/-10 in loss',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_accept_post_train     |paste - - - -" u 8:(1./$2):10 w xerror t 'FTHMC w/ 2-step stout $c=0.04287$ from average of trained model at $\beta=6$',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c14_accept_post_train |paste - - - -" u 8:(1./$2):10 w xerror t 'FTHMC w/ 2-step stout $c=0.14$',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c26_accept_post_train |paste - - - -" u 8:(1./$2):10 w xerror t 'FTHMC w/ 2-step stout $c=0.26$'

set origin orig,0
set size orig_next-orig,1
orig=orig_next
orig_next=orig_next+figw

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
	"<paste - - <../../u1_2d/stats//s_nthmc_l64_b6_5cn8_5cn10_lr1e-5_stepsize_train |awk '$2>=5&&$2<=7'" u 4:(1./$2) w lp t '',\
	"<paste - - <../../u1_2d/stats/s_fthmc_l64_b6_stepsize_train     |awk '$2>=5&&$2<=7'" u 4:(1./$2) w lp t '',\
	"<paste - - <../../u1_2d/stats/s_fthmc_l64_b6_c14_stepsize_train |awk '$2>=5&&$2<=7'" u 4:(1./$2) w lp t '',\
	"<paste - - <../../u1_2d/stats/s_fthmc_l64_b6_c26_stepsize_train |awk '$2>=5&&$2<=7'" u 4:(1./$2) w lp t ''

set origin orig,0
set size orig_next-orig,1
orig=orig_next
orig_next=orig_next+figw

set xlabel '$\gamma_{\text{HMC}}(\delta=16)\Big/\gamma_{\text{NTHMC}}(\delta=16)$'
set xrange [0.7:6.3]
set xtics 1
plot\
	"<for b in 5 6 7;do ./get_dq2_hmc_scaled_b_v_d $b 64 4; ./get_dq2_hmc_scaled_b_v_d $b 64 4 s_nthmc_l64_b5;done|paste - -" u ($5/$2):(1./$1):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w xerror lt 2 t '',\
	"<(./get_dq2_hmc_scaled_b_v_d 5 64 4; ./get_dq2_hmc_scaled_b_v_d 5 64 4 s_nthmc_l64_b6;./get_dq2_hmc_scaled_b_v_d 6 64 4; ./get_dq2_hmc_scaled_b_v_d 6 64 4 s_nthmc_l64_b6_startb6;./get_dq2_hmc_scaled_b_v_d 7 64 4; ./get_dq2_hmc_scaled_b_v_d 7 64 4 s_nthmc_l64_b6_startb7)|paste - -" u ($5/$2):(1./$1):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w xerror lt 3 t '',\
	"<for b in 5 6 7;do ./get_dq2_hmc_scaled_b_v_d $b 64 4; ./get_dq2_hmc_scaled_b_v_d $b 64 4 s_nthmc_l64_b6_5cn8_5cn10_lr1e-5;done|paste - -" u ($5/$2):(1./$1):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w xerror lt 4 t '',\
	"<for b in 5 6 7;do ./get_dq2_hmc_scaled_b_v_d $b 64 4; ./get_dq2_hmc_scaled_b_v_d $b 64 4 s_fthmc_l64_b6;done|paste - -" u ($5/$2):(1./$1):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w xerror lt 5 t '',\
	"<for b in 5 6 7;do ./get_dq2_hmc_scaled_b_v_d $b 64 4; ./get_dq2_hmc_scaled_b_v_d $b 64 4 s_fthmc_l64_b6_c14;done|paste - -" u ($5/$2):(1./$1):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w xerror lt 6 t '',\
	"<for b in 5 6 7;do ./get_dq2_hmc_scaled_b_v_d $b 64 4; ./get_dq2_hmc_scaled_b_v_d $b 64 4 s_fthmc_l64_b6_c26;done|paste - -" u ($5/$2):(1./$1):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w xerror lt 7 t ''

set origin orig,0
set size orig_next-orig,1
orig=orig_next
orig_next=orig_next+figw

set xlabel 'plaquette'
set xrange [0.89:0.93]
set xtics 0.02
plot\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_hmc_l64_params_plaq_post_train   |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b5_plaq_post_train     |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b6_plaq_post_train     |paste - - - - - - - - -|head -n1;grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b6_startb6_plaq_post_train     |paste - - - - - - - - -|head -n1;grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b6_startb7_plaq_post_train     |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b6_5cn8_5cn10_lr1e-5_plaq_post_train |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_plaq_post_train     |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c14_plaq_post_train |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c26_plaq_post_train |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t ''

set origin orig,0
set size orig_next-orig,1
orig=orig_next
orig_next=orig_next+figw

set xlabel 'plaquette transformed'
set xrange [-0.7:1.2]
set xtics 0.5
plot\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_hmc_l64_params_plaq_post_train   |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b5_plaq_post_train     |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b6_plaq_post_train     |paste - - - - - - - - -|head -n1;grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b6_startb6_plaq_post_train     |paste - - - - - - - - -|head -n1;grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b6_startb7_plaq_post_train     |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b6_5cn8_5cn10_lr1e-5_plaq_post_train |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_plaq_post_train     |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c14_plaq_post_train |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c26_plaq_post_train |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t ''


set origin orig,0
set size orig_next-orig,1
orig=orig_next
orig_next=orig_next+figw

set xlabel 'force norm'
set xrange [60:540]
set xtics 100
plot\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_hmc_l64_params_force_post_train   |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b5_force_post_train     |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b6_force_post_train     |paste - - - - - - - - -|head -n1;grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b6_startb6_force_post_train     |paste - - - - - - - - -|head -n1;grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b6_startb7_force_post_train     |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b6_5cn8_5cn10_lr1e-5_force_post_train |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_force_post_train     |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c14_force_post_train |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c26_force_post_train |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t ''

set origin orig,0
set size orig_next-orig,1
orig=orig_next
orig_next=1    # the last figure

set rmargin 2

set xlabel 'force max'
set xrange [-5:65]
set xtics 10
plot\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_hmc_l64_params_force_post_train   |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b5_force_post_train     |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b6_force_post_train     |paste - - - - - - - - -|head -n1;grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b6_startb6_force_post_train     |paste - - - - - - - - -|head -n1;grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b6_startb7_force_post_train     |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b6_5cn8_5cn10_lr1e-5_force_post_train |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_force_post_train     |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c14_force_post_train |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c26_force_post_train |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t ''

unset multiplot
