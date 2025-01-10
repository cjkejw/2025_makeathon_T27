#define BUZZER_PIN 3  // Pin connected to the buzzer
#define LED_PIN 4     // Pin connected to the LED

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(9600);  // Start Serial communication
}

void loop() {
  if (Serial.available() > 0) {
    char signal = Serial.read();  // Read data from Python

    if (signal == '1') {
      // Turn on LED
      digitalWrite(LED_PIN, HIGH);

      // Generate a fast and loud buzzing sound
      for (int i = 0; i < 20; i++) {  // Loop for 20 fast buzz cycles
        tone(BUZZER_PIN, 5000);  // Generate a 5kHz tone
        delay(20);               // ON for 20ms

        // Check if the signal has changed to '0'
        if (Serial.available() > 0 && Serial.read() == '0') {
          noTone(BUZZER_PIN);        // Turn off the buzzer immediately
          digitalWrite(LED_PIN, LOW);  // Turn off LED immediately
          break;                    // Exit the loop
        }

        noTone(BUZZER_PIN);  // Turn off the tone
        delay(20);           // OFF for 20ms
      }
    } else if (signal == '0') {
      noTone(BUZZER_PIN);         // Ensure the buzzer is OFF
      digitalWrite(LED_PIN, LOW);  // Ensure the LED is OFF
    }
  }
}