#!/bin/bash

start=$1
end=$2

for j in `seq "${start}" "${end}"` ; do
    echo "Canceling job:" $j
    scancel $j

done