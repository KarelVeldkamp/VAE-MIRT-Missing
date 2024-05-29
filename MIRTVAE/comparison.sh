#!/bin/bash

while IFS= read -r line
do
  # Read four strings from the line
  set -- $line
  arg1=$1
  arg2=$2
  arg3=$3
  arg4=$4
  # Run the Python script with the four arguments
  python3 ~/Documents/GitHub/VAE-MIRT-Missing/MIRTVAE/main.py $arg1 $arg2 $arg3 $arg4
done < simpars.txt
    
   
