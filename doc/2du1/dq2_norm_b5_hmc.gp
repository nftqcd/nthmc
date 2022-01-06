set lmargin 9
set grid lc rgb '#f0f0f0'
set key bottom
set xlabel 'MD time unit ($\delta$)'
set ylabel '$\Delta Q^2 \Big/ 2\langle Q^2\rangle_{\text{exact}}$'
pf(b,v)="<sed -n '/^beta: ".b."$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/s_hmc_l".v."_params_dq2_post_train"
nm(b,v)=2*v*v*real(system("awk '$1==".b."{print $2}' ../../u1_2d/exact_topo_sus_infv.output"))
nm64=nm("5",64)
nm128=nm("5",128)
nm256=nm("5",256)
plot\
	pf("5",64) u ($1*4):($2/nm64) w l t '$64\times 64$',\
	pf("5",128) u ($1*4):($2/nm128) w l t '$128\times 128$',\
	pf("5",256) u ($1*4):($2/nm256) w l t '$256\times 256$',\
