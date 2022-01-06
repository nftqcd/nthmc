set lmargin 9
set grid lc rgb '#f0f0f0'
set key left Left reverse
set xlabel 'MD time unit ($\delta$)'
set ylabel '$\Delta Q^2 \Big/ 2\langle Q^2\rangle_{\text{exact}}$'
pf(b,v)="<sed -n '/^beta: ".b."$/,/^beta: /{/^BEGIN dQ2$/,/^END dQ2$/{/dQ2/d;p;};}' ../../u1_2d/stats/s_hmc_l".v."_params_dq2_post_train"
nm(b,v)=2*v*v*real(system("awk '$1==".b."*".v."*".v."/4096{print $2}' ../../u1_2d/exact_topo_sus_infv.output"))
nm64=nm("6",64)
nm66=nm("6",66)
nm68=nm("6",68)
nm70=nm("6",70)
pk(v)=sprintf('$%d\times %d$, $\beta=%.4f$',v,v,6.*v*v/4096)
plot\
	pf("6",64) u ($1*4):($2/nm64):($4/nm64) w e t pk(64),\
	pf("6.380859375",66) u ($1*4):($2/nm66):($4/nm66) w e t pk(66),\
	pf("6.7734375",68) u ($1*4):($2/nm68):($4/nm68) w e t pk(68),\
	pf("7.177734375",70) u ($1*4):($2/nm70):($4/nm70) w e t pk(70)
