#! /bin/bash

names=( "John" "Mark" "James"  "Mary" "Angie" )
echo "los nombres son: ${names[*]}" #Con el * decimos todos
echo "los nombres son: ${names[@]}" #Otra forma de escribirlo

echo "First item: ${names[0]} "

echo "Los indices: ${!names[@]} " #Así sabemos la posición.

echo "El tal de items: ${#names[@]} " #Así el total de items

echo "el último elemento es: ${names[${#names[*]}-1]}"

for name in ${names[*]}
do
    echo "El nombre es: ${name}"
done

unset names[1]
echo "Los nombres son: ${names[*]}" #Así quitamos un elemento

names[${#names[*]}]="bob"
echo "Los nombres son: ${names[*]}" #Un tippo append.

names+=("Mafe" "Ángel") #Agregamos varios
echo "Los nombres son: ${names[*]}" 