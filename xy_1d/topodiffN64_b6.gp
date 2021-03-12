#!/usr/bin/env gnuplot
set term pngcairo size 1920,1200 font "Lucida Bright OT,16"
set output ARG0[*:strlen(ARG0)-2].'png'

beta(n) = \
	n==7?'1.625' : \
	n==8?'2.25' : \
	n==9?'2.875' : \
	n==10?'3.5' : \
	n==14?'6.0' : \
	0/0

set grid

set xlabel 'Batch averaged Markov chain states'
set ylabel 'RMS topological charge difference'

plot \
	[6144:*] [0:*] \
	for [n = 14:14] \
	"<awk -F '[][,: ]*' '/^topo:/{x=0;for(i=2;i<NF;++i){qq[i]=$i;x+=(qo[i]-qq[i])**2}print sqrt(x/(NF-2)); for(i in qq)qo[i]=qq[i]}' i".n.".log" u 0:1 w l t 'β = '.beta(n)
