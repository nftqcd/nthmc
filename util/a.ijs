display=: ('_';'-')stringreplace"1 ":
besselj=: 1 : '(i.0) H. (1+m)@(_0.25&*)@*: * ^&m@-: % (!m)"_'
besseli=: 1 : '(i.0) H. (1+m)@(0.25&*)@*: * ^&m@-: % (!m)"_'
plaq2dU1 =: 1 besseli % 0 besseli
errdisplay=:(display@{. , ' +/- ', display@{:)"1

NB. Computes mean and standard error of the mean from data
NB. of independent samples in the first dimension.
meanStderr=:(] ,. [: %:@:(+/@:*: % # * <:@#) [ -"_1 _ ]) (+/%#)

NB. Get the numerical results from the output of the shell command (y).
NB. Requires a verb (u) to get the numbers parsed (_&".).
NB. Optional regex replacement (x).
getRes=: 1 :0
u@:(_&".);._2 [2!:0 y
:
u@:(_&".);._2 x rxrplc [2!:0 y
)

NB. Following Wolff, 2004, and equation numbers refer to the paper.
NB. Input data is a vector of replica of boxed matrices.
NB. Each matrix consists rows of primary observables from succesive configurations.
NB. Input must have rank 1 (1 = $@$)

NB. output: Two items, the mean and the standard error of the mean from Jackknife resampling
NB. monad conjunction, m: Block size, v: Function of the observables, y: Input
NB. dyad conjunction, x: Mean, m: as in monad, v: as in monad, y: as in monad
jackknifeEst =: 2 :0
(v y) m jackknifeEst v y
:
pack=. ,@:<@:,.^:(0=L.y)
unpack=. ,@:>^:(0=L.y)
x ,: %:(+/@:*: * <:@# % #) x -"_ _1 ; (<@:(m jackknifeSampleReplica (v@:unpack f.))"_ 0 i.@:#) pack y
)
jackknifeSampleReplica =: 2 :0
:
(-m) v@:(x y}~ <)\. > y{x
)

NB. Eq. 7
NB. output: an estimate of the ensemble mean
NB. monad, y: Input
ensembleMean =: +/@:(+/&>) % +/@:(#&>)

NB. Eq. 31
NB. output: the autocorrelation function of lag x
NB. dyad, x: tLag (a scalar integer), y: Input (subtracted to be zero mean).  FIXME: add cross terms
autocorrelation =: 4 :'((+/#&>y) - x*#y) %~ +/ +/@:((-x)"_ }. x&|.*])&> y'

NB. The value of tau(w) if tau_int(w) <: 0.5
tauwMin =: 1e_4

NB. The factor S in Eq. 50, >:&1 *. <:&2
SWolff =: 1.5

NB. output: boxed items of the following
NB.   a table of 2 3=$
NB.     {. is the window determined by 0 > w
NB.     {: is the window determined by 0 > g w, Wolff, 2004
NB.     The three column correspond to
NB.       automatic selected window
NB.       integrated autocorrelation length of w
NB.       the corresponding errors
NB.   an array of values of the autocorrelation function, up to 2*w
NB. monad adverb, m: observable index (column number in y), y: Input (subtracted to be zero mean).
NB. FIXME: suport functions of observables
intautocorrelation =: 1 :0
y =. (,@:<@:,.)^:(-.@:L.)y
d =. m&{"1 &. > y
tmax =. 2#<./ #&> y
N =. +/#&> y
twWolff =. tw =. 0
tauintw =. 0.5
acs =. 0 autocorrelation d
tlag =. 1
while. tlag +./@:< tmax do.
	acs =. acs, c =. tlag autocorrelation d
	if. (0 > c) *. (0 = tw) do.
		tmax =. 0}&tmax +:tlag
		tw =. tlag
	end.
	tauw =. tauwMin"_`(SWolff % [:^. (1++:)%(_1++:))@.(>&0.5) tauintw =. tauintw + c % {.acs
	if. (0 > tlag (^@:-@:% - ] % N %:@:* [) tauw) *. (0 = twWolff) do.
		tmax =. 1}&tmax +:tlag
		twWolff =. tlag
	end.
	tlag=.>:tlag
end.
tauint =. (tw,twWolff) (0.5 -~ +/@:{.  % {.@:])"0 _ acs
tauinterr =. (* [: %: N %~ 4*0.5+tw"_) {.tauint
tauinterr =. tauinterr, (* [: %: N %~ 4*0.5+twWolff-]) {:tauint
((tw,twWolff),.tauint,.tauinterr);acs
)
