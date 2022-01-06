set grid lc rgb '#f0f0f0'

set tmargin 2
set bmargin 0
set lmargin 10
set rmargin 3

set log x
set xrange [0.134:0.266]
set multiplot layout 2,1

set key left Left reverse bottom box samplen 1 width +4
set xtics ('' 1./4, '' 1./5, '' 1./6, '' 1./7)
set ylabel 'step size'
set yrange [0.09:0.27]
plot\
	"<paste - - <../../u1_2d/stats/s_hmc_l64_params_stepsize_train   |awk '$2>=4&&$2<=7'" u (1./$2):4 w lp t '$64^2$, HMC',\
	"<paste - - <../../u1_2d/stats/s_nthmc_l64_b3_stepsize_train     |awk '$2>=4&&$2<=7'" u (1./$2):4 w lp t '$64^2$, NTHMC trained at $\beta=3$',\
	"<paste - - <../../u1_2d/stats/s_nthmc_l64_b4_cn8_stepsize_train |awk '$2>=4&&$2<=7'" u (1./$2):4 w lp t '$64^2$, NTHMC trained at $\beta=4$',\
	"<paste - - <../../u1_2d/stats/s_nthmc_l64_b5_stepsize_train     |awk '$2>=4&&$2<=7'" u (1./$2):4 w lp t '$64^2$, NTHMC trained at $\beta=5$',\
	"<paste - - <../../u1_2d/stats/s_nthmc_l64_b5_cn10_stepsize_train|awk '$2>=4&&$2<=7'" u (1./$2):4 w lp t '$64^2$, NTHMC trained at $\beta=5$'

set tmargin 0
set bmargin 4

unset key
set xtics ('$1/4$' 1./4, '$1/5$' 1./5, '$1/6$' 1./6, '$1/7$' 1./7)
set xlabel '$1/\beta$'
set ylabel 'acceptance'
set yrange [0.73:0.87]
plot\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_hmc_l64_params_accept_post_train   |paste - - - -" u (1./$2):8:10 w e t '$64^2$, HMC',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b3_accept_post_train     |paste - - - -" u (1./$2):8:10 w e t '$64^2$, NTHMC trained at $\beta=3$',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b4_cn8_accept_post_train |paste - - - -" u (1./$2):8:10 w e t '$64^2$, NTHMC trained at $\beta=4$',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b5_accept_post_train     |paste - - - -" u (1./$2):8:10 w e t '$64^2$, NTHMC trained at $\beta=5$',\
	"<grep -Ev '^(#|$)' ../../u1_2d/stats/s_nthmc_l64_b5_cn10_accept_post_train|paste - - - -" u (1./$2):8:10 w e t '$64^2$, NTHMC trained at $\beta=5$'

unset multiplot
