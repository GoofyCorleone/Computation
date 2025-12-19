const int motorpin = 2;
const int potpin = A0;
int pot;
int speed;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  pot = analogRead(potpin);
  speed = map(pot,0,1023,0,255);
  analogWrite(motorpin,speed);
  Serial.println(speed);
}
