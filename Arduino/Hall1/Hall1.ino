int led = 13; // Emplear el led integrado en el Arduino
int digitalPin = 3; // La salida digital del sensor
int analogPin = A0; // La salida análoga del sensor
int digitalVal;
int analogVal;

void setup() {
  pinMode(led,OUTPUT);
  pinMode(digitalPin,INPUT);
  Serial.begin(9600);
}

void loop() {
  digitalVal = digitalRead(digitalPin);
  if(digitalVal == HIGH)
  {
    digitalWrite(led,HIGH);
  }
  else
  {
    digitalWrite(led,LOW);
  }
  analogVal = analogRead(analogPin);
  Serial.println(analogVal);
  delay(50);
}

