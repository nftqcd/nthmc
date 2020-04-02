#!/usr/bin/env gnuplot
set term pngcairo size 1920,1200 font "Lucida Bright OT,16"
set output ARG0[*:strlen(ARG0)-2].'png'
beta='5.0'
batch=128
epoch=64
steps=2048
file='r0.log'

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

set grid

set title 'training, β=' . beta . ', N_{batch}=' . batch . ', N_{epoch}=' . epoch . ', N_{steps}=' . steps . ', ' . "`date`"
set xlabel 'β'
set ylabel 'values'
set y2label 'topological values (log scale)'

plot \
	"<awk -F '[][,: ]*' '/^accept/{x=0;for(i=2;i<NF;++i){x+=$i}print x/(NF-2)}' ".file u (steptobeta($0)):1 w p t 'batch acceptance', \
	"<awk -F '[][,: ]*' '/^topo:/{x=0;for(i=2;i<NF;++i){qq[i]=$i*$i;x+=(qo[i]-qq[i])**2}print x/(NF-2); for(i in qq)qo[i]=qq[i]}' ".file u (steptobeta($0)):1 w p t 'batch Q^2 diff', \
	"<awk -F '[][,: ]*' '/^cosDiff/{x=0;for(i=2;i<NF;++i){x+=$i}print x/(NF-2)}' ".file u (steptobeta($0)):1 w p t 'proposed cosine diff', \
	"<awk -F '[][,: ]*' '/^topoDiff/{x=0;for(i=2;i<NF;++i){x+=$i}print x/(NF-2)}' ".file u (steptobeta($0)):1 axes x1y2 w p t 'proposed approx Q^2 diff', \
	"<awk -F '[][,: ]*' '/^weights/{print $2}' ".file u (steptobeta($0)):1 w p t 'step length', \
	"<awk -F '[][,: ]*' 'w==1{print $2} w>0{w--} /^weights/{w=1}' ".file u (steptobeta($0)):(atan($1)/pi) w p t 'coef transform 1', \
	"<awk -F '[][,: ]*' 'w==1{print $2} w>0{w--} /^weights/{w=2}' ".file u (steptobeta($0)):(atan($1)/pi) w p t 'coef transform 2', \
	"<awk -F '[][,: ]*' 'w==1{print $2} w>0{w--} /^weights/{w=16}' ".file u (steptobeta($0)):(atan($1)/pi) w p t 'coef transform 16', \
	"<awk -F '[][,: ]*' '/^plaq/{x=0;for(i=2;i<NF;++i){x+=$i}print x/(NF-2)}' ".file u (steptobeta($0)):1 w p t 'plaquette', \
	plaqStep w l t 'plaquette exact'
