#! /bin/bash

function sayHello()
{
    echo "Holi a todos"
}

sayHello

function HolaA()
{
    echo "Hola, $1"
}

HolaA "Angie"

function cumple()
{
    echo "Hola yo soy $1, tengo $2 años"
}
cumple "Jafert" 23