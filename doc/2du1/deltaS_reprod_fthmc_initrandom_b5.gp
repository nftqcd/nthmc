set grid lc rgb '#f0f0f0'
set xlabel 'trajectory'
set ylabel '$\delta S$'
set log y
plot "<bash -c 'paste <(grep ^V-old ../../u1_2d/reprod/t_fthmc_0_b5_cpu0_N.log) <(grep ^V-old ../../u1_2d/reprod/t_fthmc_0_b5_cpu1_N.log) <(grep ^V-old ../../u1_2d/reprod/t_fthmc_0_b5_cuda_N.log) <(grep ^V-old ../../u1_2d/reprod/t_fthmc_0_b5_mac_N.log)'" u 0:(abs($4-$2)) w lp pt 4 t 'cpu1-cpu0', '' u 0:(abs($6-$2)) w lp pt 6 t 'cuda-cpu0', '' u 0:(abs($8-$2)) w lp pt 8 t 'mac_cpu-cpu0'
