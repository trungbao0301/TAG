//Necessary Libraries
#include <BH1750.h>
#include <Wire.h>
#include <Servo.h>
#include <avr/wdt.h>

BH1750 lightSensor;
Servo relay;

float lux = 0;
int SOLENOID_PIN = 14;
const float FIRE_LUX_THRESHOLD = 90.0;
const float LUX_RELEASE_THRESHOLD = 110.0;
const unsigned long LUX_DEBOUNCE_MS = 200UL;
const unsigned long FALLBACK_START_MS = 60000UL;
const unsigned long FALLBACK_INTERVAL_MS = 5000UL;


unsigned long now;
unsigned long low_lux_since_ms = 0;
unsigned long last_lux_run_ms = 0;
unsigned long last_fallback_run_ms = 0;
unsigned long lux_pulse_count = 0;
unsigned long fallback_pulse_count = 0;
bool auto_run_enabled = true;
bool lux_trigger_armed = true;
bool have_lux_run = false;
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

  updateAutoLux(now);

  // Stay connected indefinitely. The host bridge has an ACK watchdog that
  // reopens and probes serial if communication becomes unresponsive.
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
    Serial.println("ok pulse");
  } else if (strcmp(cmd, "fire") == 0 || strcmp(cmd, "test") == 0) {
    pulseSolenoid();
    Serial.println("ok pulse");
  } else if (strcmp(cmd, "stop") == 0) {
    relay.writeMicroseconds(1000);
    Serial.println("ok stop");
  } else if (strcmp(cmd, "auto on") == 0) {
    auto_run_enabled = true;
    low_lux_since_ms = 0;
    Serial.println("ok auto on");
  } else if (strcmp(cmd, "auto off") == 0) {
    auto_run_enabled = false;
    low_lux_since_ms = 0;
    relay.writeMicroseconds(1000);
    Serial.println("ok auto off");
  } else if (strcmp(cmd, "status") == 0) {
    lux = lightSensor.readLightLevel();
    Serial.print("ok lux=");
    Serial.print(lux);
    Serial.print(" auto=");
    Serial.print(auto_run_enabled ? "on" : "off");
    Serial.print(" armed=");
    Serial.print(lux_trigger_armed ? "yes" : "no");
    Serial.print(" fallback=");
    Serial.print(
      have_lux_run &&
      (unsigned long)(millis() - last_lux_run_ms) >= FALLBACK_START_MS
        ? "on" : "off"
    );
    Serial.print(" lux_count=");
    Serial.print(lux_pulse_count);
    Serial.print(" fallback_count=");
    Serial.println(fallback_pulse_count);
  } else {
    Serial.print("unknown ");
    Serial.println(cmd);
  }
}

void updateAutoLux(unsigned long sample_ms) {
  if (!auto_run_enabled) {
    low_lux_since_ms = 0;
    return;
  }

  if (lux > LUX_RELEASE_THRESHOLD) {
    lux_trigger_armed = true;
    low_lux_since_ms = 0;
  } else if (lux < FIRE_LUX_THRESHOLD && lux_trigger_armed) {
    if (low_lux_since_ms == 0) {
      low_lux_since_ms = sample_ms;
    }
    if ((unsigned long)(sample_ms - low_lux_since_ms) >= LUX_DEBOUNCE_MS) {
      pulseSolenoid();
      unsigned long fired_ms = millis();
      last_lux_run_ms = fired_ms;
      last_fallback_run_ms = fired_ms;
      have_lux_run = true;
      lux_pulse_count++;
      lux_trigger_armed = false;
      low_lux_since_ms = 0;
      Serial.print("event lux pulse lux=");
      Serial.println(lux);
    }
  } else {
    low_lux_since_ms = 0;
  }

  if (
    have_lux_run &&
    (unsigned long)(sample_ms - last_lux_run_ms) >= FALLBACK_START_MS &&
    (unsigned long)(sample_ms - last_fallback_run_ms) >= FALLBACK_INTERVAL_MS
  ) {
    pulseSolenoid();
    last_fallback_run_ms = millis();
    fallback_pulse_count++;
    Serial.print("event fallback pulse lux=");
    Serial.println(lux);
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
