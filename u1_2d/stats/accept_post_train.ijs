#!/usr/bin/env jconsole
load'../../util/a.ijs'

NB. naive choice of block size
bsAcc=:32

betaForFile=: 4 :0
	betastr=.display x
	echo 'beta: ',betastr
	acc=.}.getRes'9 awk ''/^beta:/{if(b==1)exit;b=0;p=0} /^beta: ',betastr,'$/{b=1} b==1&&/^# post-training inference step/{p=1} p==1&&/^accept:/{print;p=0}'' ',y
	nconf=.#acc
	13!:8&1@:('Incorrect number of configurations: ',display@:])`[ @. (nconf=]) #pmap
	echo 'nconf: ',display nconf
	echo 'blocksize: ',display bsAcc
	echo 'accept: ',errdisplay , bsAcc jackknifeEst ensembleMean acc
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
