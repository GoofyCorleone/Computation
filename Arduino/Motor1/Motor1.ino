const int Motorpin = 2;
const int butpin = 7;
int state;

void setup() {
  // put your setup code here, to run once:
  pinMode(Motorpin,OUTPUT);
  pinMode(butpin,INPUT_PULLUP);
}

void loop() {
  // put your main code here, to run repeatedly:
  state = digitalRead(butpin);
  if(state == HIGH){
    digitalWrite(Motorpin,LOW);
  }
  else{
    digitalWrite(Motorpin,HIGH);
  }
}
