#! /bin/bash

echo 10+20 #No lo interpretará
echo $(( 10+20 )) #Así con cada operador aritmético

x=10
y=20
echo $(( x % y )) #Operación módulo me da el residuo.

echo $(expr $x \* $y) #Lo mismo dice evaluar expresión