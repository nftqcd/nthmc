# for b in 1 4 16 64;do echo -n $b; awk '/^[1-9]\./ {t=$1;a=0} /jit run/{++a;s[a]=$(NF-1);if(a==6){m=10000;for(k in s){if(m>s[k])m=s[k]};printf(" %.6f",m)}}' ../../bench/omelyan_8_8_8_16_$b.log; echo; done
$T << E
1 0.118519 0.206142 0.119662 0.432068 0.752029 0.429256
4 0.425440 0.709467 0.423991 0.653284 1.118852 0.660322
16 2.394018 3.584059 2.399902 1.702473
64 9.497128 14.213241 9.533869
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
