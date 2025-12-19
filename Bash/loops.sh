#! /bin/bash

number=0

while  [ $number -lt 10 ]
do
    echo $number
    number=$((number + 1))
done

number=0
until  [ $number -ge 10 ] #Este empieza cuando la condición es falsa.
do
    echo $number
    number=$((number + 1))
done

for i in 1 2 3 4 5
do
    echo $i
done

for j in {0..100..10} #Vaya de 0 a 100 en pasos de 10 uwu
do
    echo $j
done

for (( i=0; i < 10; i++ )) #i+=20 cuánto incrementar
do
    echo $i
done