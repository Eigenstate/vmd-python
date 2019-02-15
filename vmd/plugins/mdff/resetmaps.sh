#!/bin/bash
rm -rf output/
mkdir output
for i in `seq 0 5`;
do
  mkdir output/$i
  rm -f $i.dx
  ln -s initialmaps/$i.dx $i.dx
done  
