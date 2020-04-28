#!/usr/bin/env gnuplot
set term pngcairo size 1920,1200 font "Lucida Bright OT,16"
set output ARG0[*:strlen(ARG0)-2].'png'

step(n) = \
	n==11?'10' : \
	n==12?'20' : \
	n==13?'30' : \
	0/0

set grid

set xlabel 'Batch averaged Markov chain states'
set ylabel 'RMS topological charge difference'

plot \
	[256:*] [0:*] \
	for [n = 11:13] \
	"<awk -F '[][,: ]*' '/^topo:/{x=0;for(i=2;i<NF;++i){qq[i]=$i;x+=(qo[i]-qq[i])**2}print sqrt(x/(NF-2)); for(i in qq)qo[i]=qq[i]}' i".n.".log" u 0:1 w l t 'steps/traj = '.step(n)
