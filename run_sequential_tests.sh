#!/bin/sh

for file in out/build/test/**/*Test.out; do
    ./"$file"
done
