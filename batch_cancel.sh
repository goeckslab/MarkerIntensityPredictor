#!/bin/bash

start=$1
end=$2

for j in `seq "${start}" "${end}"` ; do
    scancel $j
    echo  $j
done