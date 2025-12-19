#! /bin/bash

echo "Escoge entre elvalor 1 o 2: "
read valor

case $valor in 
    1)
        echo "tu escogiste el número 1"
        ;;
    2)
        echo "Tu escogiste el número 2"
        ;;
    *)
        echo "Valor incorrecto"
        ;;
esac

# Case es solo para valores no aritméticos