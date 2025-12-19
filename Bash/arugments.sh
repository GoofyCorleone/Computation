#! /bin/bash

echo $1 $2 $3 $4 #La separación de argumentos está dado por un espacio
echo $@ #Este lee desde el primero en adelante
echo $# #Este cuenta todos los elementos    

args=("$@")
echo "results: ${args[0]} ${args[1]} ${args[3]}"