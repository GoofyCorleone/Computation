#! /bin/bash

echo "Enter a name"
read name

echo "Write an adjective"
read adjective

result="$name is $adjective"
echo $result

echo ${name,,} #convierte todo a minúsculas las dos comas
echo ${adjective^^} #Convierte todo a mayúsculas
echo ${name,,[AEIOU]} #Convierte solo vocales a minuscula
echo ${name^^[aeiou]} #Convierte solo vocales a mayuscula