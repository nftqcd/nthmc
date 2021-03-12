#!/usr/bin/env gnuplot
set term pngcairo size 1920,1200 font "Lucida Bright OT,16"
set output ARG0[*:strlen(ARG0)-2].'png'

set grid

set xlabel 'trajectories'
set ylabel 'topological charge (shfted by multiples of 10)'

plot \
	[256:*] \
	for [n=1:4] "<sed -n '/^topo: /s/.*\\[\\(.*\\)\\]/\\1/p' i11.log" u 0:(column(n)+10*(n-1)) w l t ''
