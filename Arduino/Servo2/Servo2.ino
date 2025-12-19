#include <Servo.h>

Servo myservo;
int pos = 0;
int SPEED = 0;
bool shouldRun = true;
void setup() {
  myservo.attach(2);
  
}

void loop() {
  if (shouldRun)
  {
    myservo.write(0); // Mueve el servo a la posición 0 grados (máxima velocidad en el sentido horario)
  delay(2000); // Mantiene el servo en esa posición por 2 segundos

  myservo.write(180); // Mueve el servo a la posición 180 grados (máxima velocidad en sentido antihorario)
  delay(2000); // Mantiene el servo en esa posición por 2 segundos
  shouldRun = false;
  }
}
