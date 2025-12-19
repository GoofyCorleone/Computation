#! /bin/bash     

declare myvariable=22 #Podemos fijar el tipo de variable

declare -r pwdfile=/etc/passwd #Solo lectura variable
echo $pwdfile

pwdfile=/etc/otra #Arrojará error
