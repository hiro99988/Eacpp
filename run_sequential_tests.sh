#!/bin/sh

for file in out/build/tests/**/*Test.out; do
    ./"$file"
done
