#! /bin/bash    

age=30 #The space matters a lot

if [ $age -eq 10 ] #Dollar simbolo para interpretar variables
then
    echo "son igualeh"
fi

if [ $age -ge 5 ]
then
    echo "eh mayor o igual que 5"
fi

if [ $age -le 20 ]
then
    echo "Eh menoh o igual que 20 "
else
    echo "El número no eh mayor o igual que 20"
fi

# Para los mayor o menor que, es -gt o -lt UwU

if (( age == 30 )) # Podemos hacer lo mismo que otro lenguajes
then 
    echo "El número es igual a 30"
fi

age=19

if (( $age > 18 )) # $ parece ser innecesario en variables aritméticas
then
    echo "eres un adulto"
elif (( $age >= 17 ))
then
    echo "Eres casi un adulto"
else
    echo "Eres un niño"
fi