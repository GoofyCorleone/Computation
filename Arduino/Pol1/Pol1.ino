int led = 13; // Emplear el LED integrado en el Arduino
int digitalPin = 3; // La salida digital del sensor
volatile int pulseCount = 0; // Contador de pulsos (revoluciones)
unsigned long lastTime = 0;
float rpm = 0;

void setup() {
  pinMode(led, OUTPUT);
  pinMode(digitalPin, INPUT);
  attachInterrupt(digitalPin - 2, countPulse, RISING); // Interrupción para contar pulsos
  Serial.begin(9600);
}

void loop() {
  unsigned long currentTime = millis();
  if (currentTime - lastTime >= 1000) { // Calcular RPM cada segundo
    noInterrupts(); // Detener las interrupciones mientras se calcula
    rpm = (pulseCount * 60.0); // Convertir a RPM
    pulseCount = 0; // Reiniciar contador de pulsos
    lastTime = currentTime; // Actualizar el tiempo
    interrupts(); // Reanudar interrupciones
    Serial.print("RPM: ");
    Serial.println(rpm);
  }
}

void countPulse() {
  pulseCount++; // Incrementar contador de pulsos
}