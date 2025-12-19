#! /bin/bash

#Estandar output es cuando da un resultado y std erro es cuando saca error

ls 1>file.txt  #guardamos todo lo que esté acá
ls -123 #Y ahora aquí lo limpia pq este comando no existe
ls -123 1>file.txt 2>errors.txt #Me guarda el erro en el otro archivo
ls >archivo.txt 2>&1 #Me guarda el output y erro en mismo archivo
ls -123 >& file2.txt #Lo mismo de arriba