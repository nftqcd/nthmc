set lmargin 9
set grid lc rgb '#f0f0f0'
set key left Left reverse bottom samplen 1
set xlabel 'MDTU ($\delta$)'
set ylabel '$\Gamma_t(\delta)$'
v='64'
pf(b)="<sed -n '/^beta: ".b."$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/s_hmc_l".v."_params_dq2_post_train"
nm(b)=2*real(v)*real(v)*real(system("awk '$1==".b."{print $2}' ../../u1_2d/exact_topo_sus_infv.output"))
nm4=nm("4")
nm5=nm("5")
nm6=nm("6")
nm7=nm("7")
plot\
	pf("4") u ($1*4):(1.-$2/nm4):($4/nm4) w e t '$\beta=4$',\
	pf("5") u ($1*4):(1.-$2/nm5):($4/nm5) w e t '$\beta=5$',\
	pf("6") u ($1*4):(1.-$2/nm6):($4/nm6) w e t '$\beta=6$',\
	pf("7") u ($1*4):(1.-$2/nm7):($4/nm7) w e t '$\beta=7$'
