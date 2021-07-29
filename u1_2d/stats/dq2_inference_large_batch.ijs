#!/usr/bin/env jconsole
load'../../util/a.ijs'

dq2=: 4 :'(+/%#)@:((-x)"_ }. x&|.*:@:-])&> y'"0 _
ds=:i.16

betaForFile=: 4 :0
	betastr=.display x
	echo 'beta: ',betastr
	topo=.]getRes'9 awk ''/^beta:/{if(b==1)exit;b=0;p=0} /^beta: ',betastr,'$/{b=1} b==1&&/^# inference step/{p=1} p==1&&/^topo:/{sub(".*: +","");gsub("[[\\]]","");print;p=0}'' ',y
	echo (>'nconf: ';'nbatch: '),.display ,.$topo
	echo 'BEGIN dQ2'
	echo (display,.ds),.' ',.errdisplay meanStderr |: ds dq2 <"1|:topo
	echo 'END dQ2'
)

resultsForFile=: 3 :0
	es=.display@{. , ' +/- ', display@{:
	echo ''
	echo '# File: ',y
	betas=.}.@:(_&".);._2[2!:0'grep ''^beta: '' ',y
	betas betaForFile"0 _ y
)

ARGV=.2}.ARGV
resultsForFile&> ARGV
exit''