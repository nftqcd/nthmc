#!/bin/sh
if grep -q 'done training epoch' "$@";then
	awk '/^beta: /{print} /^weights: /{gsub("[[\\]]","");w=$2} /done training epoch/{print "stepsize:",w}' "$@"
elif grep -q 'done step tuning' "$@";then
	awk '/^beta: /{print} /done step tuning/{p=1} /^using .* dt /&&p{print "stepsize:",$4,"step/traj:",$6;p=0}' "$@"
else
	echo "unknown log file: '$@'" >&2
	exit 1
fi
