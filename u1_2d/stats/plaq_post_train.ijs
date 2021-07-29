#!/usr/bin/env jconsole
load'../../util/a.ijs'

NB. naive choice of block size
bsPlaq=:32

betaForFile=: 4 :0
	betastr=.display x
	echo 'beta: ',betastr
	plaq=.}.getRes'9 awk ''/^beta:/{if(b==1)exit;b=0;p=0} /^beta: ',betastr,'$/{b=1} b==1&&/^# post-training inference step/{p=1} p==1&&/^plaq:/{print;p=0}'' ',y
	pmap=.}.getRes'9 awk ''/^beta:/{if(b==1)exit;b=0;p=0} /^beta: ',betastr,'$/{b=1} b==1&&/^# post-training inference step/{p=1} p==1&&/^plaqWoTrans:/{print;p=0}'' ',y
	nconf=.#plaq
	13!:8&1@:('Incorrect number of configurations: ',display@:])`[ @. (nconf=]) #pmap
	echo 'nconf: ',display nconf
	echo 'blocksize: ',display bsPlaq
	echo (>'plaq: ';'plaqMin: ';'plaqMax: '),.errdisplay |: bsPlaq jackknifeEst ensembleMean ,<plaq
	echo (>'plaqMapped: ';'plaqMappedMin: ';'plaqMappedMax: '),.errdisplay |: bsPlaq jackknifeEst ensembleMean ,<pmap
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
