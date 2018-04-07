#!/bin/bash

echo "running tests in pics/"

for i in `ls pics/`
do
    echo "pics/$i"
    python3 main.py "pics/$i" --all

    #python3 main.py pics/"$i" --all --label
    echo "done"
    read varname

done


