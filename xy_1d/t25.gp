#!/usr/bin/env gnuplot
set term pngcairo size 1920,1200 font "Lucida Bright OT,16"
set output ARG0[*:strlen(ARG0)-2].'png'
beta='3.5'
batch=1024
epoch=4
stepsTrain=512
stepsMix=64

steps=stepsTrain+stepsMix
file=ARG0[*:strlen(ARG0)-2].'log'

j(ex)="<jconsole -js 'exit echo ".ex."'"
$plaq<<e
(,: 1 bi % 0 bi=:1 : '"'"'(i.0) H. (1+m)@(0.25&*)@*: * ^&m@-: % (!m)"_'"'"')
e
plaqStep=j('($~2,~2%~#),|:(,~(e%~b,0)-~])' . $plaq[1] . '>:(b=:'.beta.'-1)*e%~>:i.e=:'.epoch)
steptobeta(n)=1.+(beta-1.)*n/steps/epoch
steptobetaT(n)=1.+(beta-1.)*((1+int(n/stepsTrain))*stepsMix+n)/steps/epoch
betatostep(b) = (b-1.)*steps*epoch/(beta-1.)
step(n) = n
stepT(n) = n+stepsMix

set pointsize 0.25

set sample steps
set log y2
set ytics nomirror
set y2tics nomirror
set autoscale  y
set autoscale y2
set grid

set title 'training, β=' . beta . ', N_{batch}=' . batch . ', N_{epoch}=' . epoch . ', N_{stepsTrain}=' . stepsTrain . ', N_{stepsMix}='. stepsMix . ', ' . system('date -r '.file)
set xlabel 'β'
#set xlabel 'training steps'
set ylabel 'values'
set y2label 'topological values (log scale)'

do for [i=1:20] { eval 'st'.i.'="w p lc '.i.' pt '.i.' notitle, 1/0 w p lc '.i.' pt '.i.' lw 1.5 ps 1.5 t"' }
plot \
	"<awk -F '[][,: ]*' '/^accept/{print $2}' ".file u (steptobeta($0)):1 @st1 'batch acceptance', \
	"<awk -F '[][,: ]*' '/^topo:/{x=0;for(i=2;i<NF;++i){qq[i]=$i;x+=(qo[i]-qq[i])**2}print x/(NF-2); for(i in qq)qo[i]=qq[i]}' ".file u (steptobeta($0)):1 axes x1y2 @st2 'batch Q diff^2', \
	"<awk -F '[][,: ]*' '/^topoDiff/{print $2}' ".file u (steptobeta($0)):1 axes x1y2 @st3 'proposed approx Q diff^2', \
	"<awk -F '[][,: ]*' '/^cosDiff/{print $2}' ".file u (steptobeta($0)):1 @st4 'proposed 1-cosine diff', \
	[steptobetaT(0):*] 0.5 @st5 'step length', \
	"<awk -F '[][,: ]*' '/^weights: /{print $2}' ".file u (steptobetaT($0)):(atan($1)/pi) @st6 'coef transform 1', \
	"<awk -F '[][,: ]*' 'w==1{print $2} w>0{w--} /^weights: /{w=11}' ".file u (steptobetaT($0)):(atan($1)/pi) @st7 'coef transform 12', \
	"<awk -F '[][,: ]*' '/^plaq/{print $2}' ".file u (steptobeta($0)):1 @st8 'plaquette', \
	plaqStep u 1:2 w l lc rgb '#a0a0a0' t 'plaquette exact'
