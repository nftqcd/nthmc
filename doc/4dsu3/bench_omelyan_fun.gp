# for b in 1 4 16 64;do echo -n $b; awk '/^[1-9]\./ {t=$1;a=0} /fun run/{++a;s[a]=$(NF-1);if(a==6){m=10000;for(k in s){if(m>s[k])m=s[k]};printf(" %.6f",m)}}' ../../bench/omelyan_8_8_8_16_$b.log; echo; done
$T << E
1 0.169766 0.284381 0.177041 1.168664 2.145144 1.208663
4 0.527040 0.858423 0.534549 1.435069 2.579894 1.502601
16 1.943588 3.229018 1.946642 2.855340 4.651465
64 7.510009 12.676793 7.583823 8.450016
E

set key left Left reverse
set grid lc rgb '#aaa'
set xlabel 'batch size'
set ylabel 'time (sec)'
set log xy
set xrange [0.8:80]
set xtics (1,4,16,64)
plot \
	$T u 1:2 w lp t 'MomM', \
	$T u 1:3 w lp t 'MomV', \
	$T u 1:4 w lp t 'MomV_gradM', \
	$T u 1:5 w lp t 'Part MomM', \
	$T u 1:6 w lp t 'Part MomV', \
	$T u 1:7 w lp t 'Part MomV_diffM'
