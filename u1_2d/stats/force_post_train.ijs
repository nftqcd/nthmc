#!/usr/bin/env jconsole
load'../../util/a.ijs'

NB. naive choice of block size
bs=:32

betaForFile=: 4 :0
	betastr=.display x
	echo 'beta: ',betastr
	force=.}.getRes'9 awk ''/^beta:/{if(b==1)exit;b=0;p=0} /^beta: ',betastr,'$/{b=1} b==1&&/^# post-training inference step/{p=1} p==1&&/^force:/{print;p=0}'' ',y
	nconf=.#force
	echo 'nconf: ',display nconf
	echo 'blocksize: ',display bs
	echo (>'forceNorm2: ';'forceNorm2Min: ';'forceNorm2Max: ';'forceNormInf: ';'forceNormInfMin: ';'forceNormInfMax: '),.errdisplay |: bs jackknifeEst ensembleMean ,<force
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
