//Necessary Libraries
#include <BH1750.h>
#include <Wire.h>
#include <Servo.h>
#include <avr/wdt.h>

BH1750 lightSensor;
Servo relay;

float lux = 0;
int SOLENOID_PIN = 14;
const float FIRE_LUX_THRESHOLD = 80.0;
const unsigned long AUTO_RUN_INTERVAL_MS = 60000UL;


unsigned long now;
unsigned long last_run_ms = 0;
bool auto_run_enabled = false;
char serial_cmd[16];
byte serial_cmd_len = 0;


void setup() {

  Serial.begin(9600);
  relay.attach(SOLENOID_PIN);
  // I2C Bus
  Wire.begin();

  lightSensor.begin();

  relay.writeMicroseconds(1000);
}

void loop() {
  handleSerial();

  lux = lightSensor.readLightLevel();
  now = millis();
  delay(40);
  handleSerial();
  now = millis();

  if (auto_run_enabled && (unsigned long)(now - last_run_ms) >= AUTO_RUN_INTERVAL_MS) {
    pulseSolenoid();
    last_run_ms = now;
    Serial.println("ok auto pulse");
  }

  if (now > 600000) {
    reboot();
  }
}

void handleSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (serial_cmd_len > 0) {
        serial_cmd[serial_cmd_len] = '\0';
        handleCommand(serial_cmd);
        serial_cmd_len = 0;
      }
      continue;
    }
    if (serial_cmd_len < sizeof(serial_cmd) - 1) {
      serial_cmd[serial_cmd_len++] = c;
    }
  }
}

void handleCommand(char *cmd) {
  if (strcmp(cmd, "run") == 0) {
    pulseSolenoid();
    auto_run_enabled = true;
    last_run_ms = millis();
    Serial.println("ok pulse");
  } else if (strcmp(cmd, "fire") == 0 || strcmp(cmd, "test") == 0) {
    pulseSolenoid();
    if (auto_run_enabled) {
      last_run_ms = millis();
    }
    Serial.println("ok pulse");
  } else if (strcmp(cmd, "stop") == 0) {
    auto_run_enabled = false;
    relay.writeMicroseconds(1000);
    Serial.println("ok stop");
  } else if (strcmp(cmd, "status") == 0) {
    lux = lightSensor.readLightLevel();
    Serial.print("ok lux=");
    Serial.print(lux);
    Serial.print(" auto=");
    Serial.println(auto_run_enabled ? "on" : "off");
  } else {
    Serial.print("unknown ");
    Serial.println(cmd);
  }
}

void pulseSolenoid() {
  relay.writeMicroseconds(2000);
  delay(30);
  relay.writeMicroseconds(1000);
}

void reboot() {
  wdt_disable();
  wdt_enable(WDTO_15MS);
  while (1) {}
}
