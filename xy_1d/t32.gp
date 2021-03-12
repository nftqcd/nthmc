#!/usr/bin/env gnuplot
set term pngcairo size 1920,1200 font "Lucida Bright OT,16"
set output ARG0[*:strlen(ARG0)-2].'png'
beta0='1.0'
beta='3.5'
batch=512
epoch=4
stepsTrain=1536
stepsMix=256

steps=stepsTrain+stepsMix
file=ARG0[*:strlen(ARG0)-2].'log'

j(ex)="<jconsole -js 'exit echo ".ex."'"
$plaq<<e
(,: 1 bi % 0 bi=:1 : '"'"'(i.0) H. (1+m)@(0.25&*)@*: * ^&m@-: % (!m)"_'"'"')
e
plaqStep=j('($~2,~2%~#),|:(,~(e%~b,0)-~])' . $plaq[1] .beta0.'+(b=:'.beta.'-'.beta0.')*e%~>:i.e=:'.epoch)
steptobeta(n)=beta0+(beta-beta0)*n/steps/epoch
steptobetaT(n)=beta0+(beta-beta0)*((1+int(n/stepsTrain))*stepsMix+n)/steps/epoch
#betatostep(b) = (b-beta0)*steps*epoch/(beta-beta0)
#step(n) = n
#stepT(n) = n+stepsMix

set pointsize 0.25

set log y2
set ytics nomirror
set y2tics nomirror
set autoscale  y
set autoscale y2
set grid

set title 'training size N=256, β=' . beta . ', N_{batch}=' . batch . ', N_{epoch}=' . epoch . ', N_{stepsTrain}=' . stepsTrain . ', N_{stepsMix}='. stepsMix . ', ' . system('date -r '.file)
#set xlabel 'β'
set xlabel 'β'
set ylabel 'values'
set y2label 'topological values (log scale)'

do for [i=1:20] { eval 'st'.i.'="w p lc '.i.' pt '.i.' notitle, 1/0 w p lc '.i.' pt '.i.' lw 1.5 ps 1.5 t"' }
plot \
	"<cat ".file." | awk -F '[][,: ]*' '/^accept/{print $2}' " u (steptobeta($0)):1 @st1 'batch acceptance', \
	"<cat ".file." | awk -F '[][,: ]*' '/^topo:/{x=0;for(i=2;i<NF;++i){qq[i]=$i;x+=(qo[i]-qq[i])**2}print x/(NF-2); for(i in qq)qo[i]=qq[i]}' " u (steptobeta($0)):1 axes x1y2 @st2 'batch Q diff^2', \
	"<cat ".file." | awk -F '[][,: ]*' '/^topoDiff/{print $2}' " u (steptobeta($0)):1 axes x1y2 @st3 'proposed approx Q diff^2', \
	"<cat ".file." | awk -F '[][,: ]*' '/^cosDiff/{print $2}' " u (steptobeta($0)):1 @st4 'proposed 1-cosine diff', \
	"<cat ".file." | awk -F '[][,: ]*' '/^weights/{print $2}' " u (steptobetaT($0)):1 @st5 'step length', \
	"<cat ".file." | awk -F '[][,: ]*' 'w==1{print $2} w>0{w--} /^weights/{w=1}' " u (steptobetaT($0)):(atan($1)/pi) @st6 'coef transform 1', \
	"<cat ".file." | awk -F '[][,: ]*' 'w==1{print $2} w>0{w--} /^weights/{w=12}' " u (steptobetaT($0)):(atan($1)/pi) @st7 'coef transform 12', \
	"<cat ".file." | awk -F '[][,: ]*' '/^plaq/{print $2}' " u (steptobeta($0)):1 @st8 'plaquette', \
	plaqStep w l lc rgb '#a0a0a0' t 'plaquette exact'
