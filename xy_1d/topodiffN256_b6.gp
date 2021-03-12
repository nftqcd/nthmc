#!/usr/bin/env gnuplot
set term pngcairo size 1920,1200 font "Lucida Bright OT,16"
set output ARG0[*:strlen(ARG0)-2].'png'

step(n) = \
	n==11?'10' : \
	n==12?'20' : \
	n==13?'30' : \
	n==15?'10' : \
	n==16?'20' : \
	n==17?'30' : \
	0/0

set grid

set xlabel 'Batch averaged Markov chain states'
set ylabel 'RMS topological charge difference (shifted by multiples of 0.2)'

plot \
	[12288:*] [0:*] \
	for [n = 15:17] \
	"<awk -F '[][,: ]*' '/^topo:/{x=0;for(i=2;i<NF;++i){qq[i]=$i;x+=(qo[i]-qq[i])**2}print sqrt(x/(NF-2)); for(i in qq)qo[i]=qq[i]}' i".n.".log" u 0:($1+0.2*(n-15)) w l t 'steps/traj = '.step(n)
