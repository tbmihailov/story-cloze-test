#!/usr/bin/env bash

PATTERN=.
if [ -n "$1" ]
then
  PATTERN=$1
fi     # $String is null.

for i in $PATTERN; do
    echo "${i}"
    python tools/parse_results_epochs_v1.py $i
done
