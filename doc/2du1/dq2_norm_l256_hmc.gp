set lmargin 9
set grid lc rgb '#f0f0f0'
set xlabel 'MD time unit ($\delta$)'
set ylabel '$\Delta Q^2 \Big/ 2\langle Q^2\rangle_{\text{exact}}$'
v='256'
pf(b)="<sed -n '/^beta: ".b."$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/s_hmc_l".v."_params_dq2_post_train"
nm(b)=2*real(v)*real(v)*real(system("awk '$1==".b."{print $2}' ../../u1_2d/exact_topo_sus_infv.output"))
nm1=nm("1")
nm2=nm("2")
nm3=nm("3")
nm4=nm("4")
nm5=nm("5")
nm6=nm("6")
nm7=nm("7")
plot\
	pf("1") u ($1*4):($2/nm1):($4/nm1) w e t '$\beta=1$',\
	pf("2") u ($1*4):($2/nm2):($4/nm2) w e t '$\beta=2$',\
	pf("3") u ($1*4):($2/nm3):($4/nm3) w e t '$\beta=3$',\
	pf("4") u ($1*4):($2/nm4):($4/nm4) w e t '$\beta=4$',\
	pf("5") u ($1*4):($2/nm5):($4/nm5) w e t '$\beta=5$',\
	pf("6") u ($1*4):($2/nm6):($4/nm6) w e t '$\beta=6$',\
	pf("7") u ($1*4):($2/nm7):($4/nm7) w e t '$\beta=7$'
