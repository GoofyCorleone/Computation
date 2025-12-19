#! /bin/bash

age=18

#18 < age < 40

if [ $age -gt 18 ] && [ $age -lt 40 ]
then
    echo "Edad válida"
else
    echo "Edad no válida"
fi

age=20

if [[ $age -gt 18 && $age -lt 40 ]] # Otra sintaxis
then
    echo "Edad válida"
else
    echo "Edad no válida"
fi

age=17

if [[ $age -gt 18 || $age -lt 40 ]] # Para el or sintaxis
then
    echo "Edad válida"
else
    echo "Edad no válida"
fi
