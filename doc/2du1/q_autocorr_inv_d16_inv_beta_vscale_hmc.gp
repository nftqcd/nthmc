set lmargin 12
set grid lc rgb '#f0f0f0'
set xlabel '$1/\beta$'
set ylabel '$\gamma(\delta=16)$' # delta in MDTU, 4 trajectories here
set log xy
set xrange [0.095:0.26]
set yrange [1:1e5]
set xtics ('$1/4$' 1./4, '$1/5$' 1./5, '$1/6$' 1./6, '$1/7$' 1./7, '$1/8$' 1./8, '$1/9$' 1./9, '$1/10$' 1./10)
f(x) = c*x**(a+b*log(x))
c = 465.3886904
a = 12.36505171
b = 6.348620753
plot\
    f(x) w l lt 21 t '',\
	"<for v in 64 66 68 70 72 74 76;do ./get_dq2_hmc_scaled_b_v_d 4 $v 4;done" u (1./$1):(1./$2):($3/$2/$2) w e t sprintf('$V/\beta = %.2f$, HMC', 64*64/4.),\
	"<for v in 64 66 68 70 72 74 76;do ./get_dq2_hmc_scaled_b_v_d 5 $v 4;done" u (1./$1):(1./$2):($3/$2/$2) w e t sprintf('$V/\beta = %.2f$, HMC', 64*64/5.),\
	"<for v in 64 66 68 70 72 74 76;do ./get_dq2_hmc_scaled_b_v_d 6 $v 4;done" u (1./$1):(1./$2):($3/$2/$2) w e t sprintf('$V/\beta = %.2f$, HMC', 64*64/6.),\
	"<for v in 64 66 68 70 72 74 76;do ./get_dq2_hmc_scaled_b_v_d 7 $v 4;done" u (1./$1):(1./$2):($3/$2/$2) w e t sprintf('$V/\beta = %.2f$, HMC', 64*64/7.)
