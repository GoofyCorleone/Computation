#! /bin/bash

while read line
do  
    echo $line
done < "${1:-/dev/stdin}"
# El estandar input es lo que escribamos o le pasemos como archivo
