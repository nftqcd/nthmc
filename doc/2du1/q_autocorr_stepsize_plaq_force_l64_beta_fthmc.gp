set grid lc rgb '#a0a0a0'

set tmargin 11.5
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
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_accept_post_train     |paste - - - -" u 8:(1./$2):10 w xerror t 'FTHMC w/ 2-step stout $c=0.04287$ from average of trained model at $\beta=6$',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c06_accept_post_train |paste - - - -" u 8:(1./$2):10 w xerror t 'FTHMC w/ 2-step stout $c=0.06$',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c1_accept_post_train  |paste - - - -" u 8:(1./$2):10 w xerror t 'FTHMC w/ 2-step stout $c=0.1$',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c14_accept_post_train |paste - - - -" u 8:(1./$2):10 w xerror t 'FTHMC w/ 2-step stout $c=0.14$',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c18_accept_post_train |paste - - - -" u 8:(1./$2):10 w xerror t 'FTHMC w/ 2-step stout $c=0.18$',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c22_accept_post_train |paste - - - -" u 8:(1./$2):10 w xerror t 'FTHMC w/ 2-step stout $c=0.22$',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c26_accept_post_train |paste - - - -" u 8:(1./$2):10 w xerror t 'FTHMC w/ 2-step stout $c=0.26$',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c3_accept_post_train  |paste - - - -" u 8:(1./$2):10 w xerror t 'FTHMC w/ 2-step stout $c=0.3$'

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
	"<paste - - <../../u1_2d/stats/s_fthmc_l64_b6_stepsize_train     |awk '$2>=5&&$2<=7'" u 4:(1./$2) w lp t '',\
	"<paste - - <../../u1_2d/stats/s_fthmc_l64_b6_c06_stepsize_train |awk '$2>=5&&$2<=7'" u 4:(1./$2) w lp t '',\
	"<paste - - <../../u1_2d/stats/s_fthmc_l64_b6_c1_stepsize_train  |awk '$2>=5&&$2<=7'" u 4:(1./$2) w lp t '',\
	"<paste - - <../../u1_2d/stats/s_fthmc_l64_b6_c14_stepsize_train |awk '$2>=5&&$2<=7'" u 4:(1./$2) w lp t '',\
	"<paste - - <../../u1_2d/stats/s_fthmc_l64_b6_c18_stepsize_train |awk '$2>=5&&$2<=7'" u 4:(1./$2) w lp t '',\
	"<paste - - <../../u1_2d/stats/s_fthmc_l64_b6_c22_stepsize_train |awk '$2>=5&&$2<=7'" u 4:(1./$2) w lp t '',\
	"<paste - - <../../u1_2d/stats/s_fthmc_l64_b6_c26_stepsize_train |awk '$2>=5&&$2<=7'" u 4:(1./$2) w lp t '',\
	"<paste - - <../../u1_2d/stats/s_fthmc_l64_b6_c3_stepsize_train  |awk '$2>=5&&$2<=7'" u 4:(1./$2) w lp t ''

set origin orig,0
set size orig_next-orig,1
orig=orig_next
orig_next=orig_next+figw

set xlabel '$\gamma_{\text{HMC}}(\delta=16)\Big/\gamma_{\text{FTHMC}}(\delta=16)$'
set xrange [0.7:7.3]
set xtics 1
plot\
	"<for b in 5 6 7;do ./get_dq2_hmc_scaled_b_v_d $b 64 4; ./get_dq2_hmc_scaled_b_v_d $b 64 4 s_fthmc_l64_b6;    done|paste - -" u ($5/$2):(1./$1):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w xerror t '',\
	"<for b in 5 6 7;do ./get_dq2_hmc_scaled_b_v_d $b 64 4; ./get_dq2_hmc_scaled_b_v_d $b 64 4 s_fthmc_l64_b6_c06;done|paste - -" u ($5/$2):(1./$1):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w xerror t '',\
	"<for b in 5 6 7;do ./get_dq2_hmc_scaled_b_v_d $b 64 4; ./get_dq2_hmc_scaled_b_v_d $b 64 4 s_fthmc_l64_b6_c1; done|paste - -" u ($5/$2):(1./$1):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w xerror t '',\
	"<for b in 5 6 7;do ./get_dq2_hmc_scaled_b_v_d $b 64 4; ./get_dq2_hmc_scaled_b_v_d $b 64 4 s_fthmc_l64_b6_c14;done|paste - -" u ($5/$2):(1./$1):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w xerror t '',\
	"<for b in 5 6 7;do ./get_dq2_hmc_scaled_b_v_d $b 64 4; ./get_dq2_hmc_scaled_b_v_d $b 64 4 s_fthmc_l64_b6_c18;done|paste - -" u ($5/$2):(1./$1):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w xerror t '',\
	"<for b in 5 6 7;do ./get_dq2_hmc_scaled_b_v_d $b 64 4; ./get_dq2_hmc_scaled_b_v_d $b 64 4 s_fthmc_l64_b6_c22;done|paste - -" u ($5/$2):(1./$1):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w xerror t '',\
	"<for b in 5 6 7;do ./get_dq2_hmc_scaled_b_v_d $b 64 4; ./get_dq2_hmc_scaled_b_v_d $b 64 4 s_fthmc_l64_b6_c26;done|paste - -" u ($5/$2):(1./$1):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w xerror t '',\
	"<for b in 5 6 7;do ./get_dq2_hmc_scaled_b_v_d $b 64 4; ./get_dq2_hmc_scaled_b_v_d $b 64 4 s_fthmc_l64_b6_c3; done|paste - -" u ($5/$2):(1./$1):($5/$2*sqrt(($3/$2)**2+($6/$5)**2)) w xerror t ''

set origin orig,0
set size orig_next-orig,1
orig=orig_next
orig_next=orig_next+figw

set xlabel 'plaquette'
set xrange [0.89:0.93]
set xtics 0.02
plot\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_plaq_post_train     |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c06_plaq_post_train |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c1_plaq_post_train  |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c14_plaq_post_train |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c18_plaq_post_train |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c22_plaq_post_train |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c26_plaq_post_train |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c3_plaq_post_train  |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t ''

set origin orig,0
set size orig_next-orig,1
orig=orig_next
orig_next=orig_next+figw

set xlabel 'plaquette transformed'
set xrange [-0.7:1.2]
set xtics 0.5
plot\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_plaq_post_train     |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c06_plaq_post_train |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c1_plaq_post_train  |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c14_plaq_post_train |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c18_plaq_post_train |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c22_plaq_post_train |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c26_plaq_post_train |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c3_plaq_post_train  |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t ''


set origin orig,0
set size orig_next-orig,1
orig=orig_next
orig_next=orig_next+figw

set xlabel 'force norm'
set xrange [50:750]
set xtics 100
plot\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_force_post_train     |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c06_force_post_train |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c1_force_post_train  |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c14_force_post_train |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c18_force_post_train |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c22_force_post_train |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c26_force_post_train |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c3_force_post_train  |paste - - - - - - - - -" u 8:(1./$2):10 w xerror t ''

set origin orig,0
set size orig_next-orig,1
orig=orig_next
orig_next=1    # the last figure

set rmargin 2

set xlabel 'force max'
set xrange [-5:85]
set xtics 10
plot\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_force_post_train     |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c06_force_post_train |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c1_force_post_train  |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c14_force_post_train |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c18_force_post_train |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c22_force_post_train |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c26_force_post_train |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t '',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_fthmc_l64_b6_c3_force_post_train  |paste - - - - - - - - -" u 20:(1./$2):22 w xerror t ''

unset multiplot
