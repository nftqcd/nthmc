#!/bin/sh
awk '/^beta: /{print} /^weights: /{gsub("[[\\]]","");w=$2} /done training epoch/{print "stepsize:",w}' "$@"
