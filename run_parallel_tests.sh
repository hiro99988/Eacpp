#!/bin/sh

for file in out/build/test/**/*TestM.out; do
    mpirun -n 4 ./"$file"
done
