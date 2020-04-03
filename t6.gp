#!/usr/bin/env gnuplot
set term pngcairo size 1920,1200 font "Lucida Bright OT,16"
set output ARG0[*:strlen(ARG0)-2].'png'
beta='6.0'
batch=512
epoch=16
steps=4096

file=ARG0[*:strlen(ARG0)-2].'log'

j(ex)="<jconsole -js 'exit echo ".ex."'"
$plaq<<e
(,: 1 bi % 0 bi=:1 : '"'"'(i.0) H. (1+m)@(0.25&*)@*: * ^&m@-: % (!m)"_'"'"')
e
plaqStep=j('($~2,~2%~#),|:(,~(e%~b,0)-~])' . $plaq[1] . '>:(b=:'.beta.'-1)*e%~>:i.e=:'.epoch)
steptobeta(n)=1.+(beta-1.)*n/steps/epoch

set pointsize 0.25

set log y2
set ytics nomirror
set y2tics nomirror
set autoscale  y
set autoscale y2
set grid

set title 'training, β=' . beta . ', N_{batch}=' . batch . ', N_{epoch}=' . epoch . ', N_{steps}=' . steps . ', ' . system('date -r '.file)
set xlabel 'β'
set ylabel 'values'
set y2label 'topological values (log scale)'

do for [i=1:20] { eval 'st'.i.'="w p lc '.i.' pt '.i.' notitle, 1/0 w p lc '.i.' pt '.i.' lw 1.5 ps 1.5 t"' }
plot \
	"<awk -F '[][,: ]*' '/^accept/{x=0;for(i=2;i<NF;++i){x+=$i}print x/(NF-2)}' ".file u (steptobeta($0)):1 @st1 'batch acceptance', \
	"<awk -F '[][,: ]*' '/^topo:/{x=0;for(i=2;i<NF;++i){qq[i]=$i;x+=(qo[i]-qq[i])**2}print x/(NF-2); for(i in qq)qo[i]=qq[i]}' ".file u (steptobeta($0)):1 axes x1y2 @st2 'batch Q diff^2', \
	"<awk -F '[][,: ]*' '/^topoDiff/{x=0;for(i=2;i<NF;++i){x+=$i}print x/(NF-2)}' ".file u (steptobeta($0)):1 axes x1y2 @st3 'proposed approx Q diff^2', \
	"<awk -F '[][,: ]*' '/^cosDiff/{x=0;for(i=2;i<NF;++i){x+=$i}print x/(NF-2)}' ".file u (steptobeta($0)):1 @st4 'proposed 1-cosine diff', \
	"<awk -F '[][,: ]*' '/^weights/{print $2}' ".file u (steptobeta($0)):1 @st5 'step length', \
	"<awk -F '[][,: ]*' 'w==1{print $2} w>0{w--} /^weights/{w=1}' ".file u (steptobeta($0)):(atan($1)/pi) @st6 'coef transform 1', \
	"<awk -F '[][,: ]*' 'w==1{print $2} w>0{w--} /^weights/{w=2}' ".file u (steptobeta($0)):(atan($1)/pi) @st7 'coef transform 2', \
	"<awk -F '[][,: ]*' 'w==1{print $2} w>0{w--} /^weights/{w=16}' ".file u (steptobeta($0)):(atan($1)/pi) @st8 'coef transform 16', \
	"<awk -F '[][,: ]*' '/^plaq/{x=0;for(i=2;i<NF;++i){x+=$i}print x/(NF-2)}' ".file u (steptobeta($0)):1 @st9 'plaquette', \
	plaqStep w l t 'plaquette exact'
