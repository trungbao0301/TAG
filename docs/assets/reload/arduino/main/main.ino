//Necessary Libraries
#include <BH1750.h>
#include <Wire.h>
#if defined(ARDUINO_ARCH_AVR)
#include <Servo.h>
#include <avr/wdt.h>
#elif !defined(ARDUINO_ARCH_ESP32)
#error "This firmware supports AVR and ESP32 boards only"
#endif

BH1750 lightSensor;
#if defined(ARDUINO_ARCH_AVR)
Servo relay;
#endif

float lux = 0;
// The RC switch signal is wired to the physical A0 header pin. On AVR Nano,
// A0 happens to equal numeric pin 14, but numeric 14 is the RGB LED on Nano
// ESP32. Use the named pin so both boards drive the same physical connector.
const int SOLENOID_PIN = A0;
const float FIRE_LUX_THRESHOLD = 90.0;
const float LUX_RELEASE_THRESHOLD = 110.0;
const unsigned long LUX_DEBOUNCE_MS = 200UL;
const unsigned long SOLENOID_ON_MS = 150UL;
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
bool light_sensor_ready = false;
byte light_sensor_address = 0;
unsigned long last_sensor_retry_ms = 0;
char serial_cmd[16];
byte serial_cmd_len = 0;

#if defined(ARDUINO_ARCH_ESP32)
const int SOLENOID_PWM_CHANNEL = 0;
const int SOLENOID_PWM_HZ = 50;
const int SOLENOID_PWM_BITS = 16;
#endif


void attachSolenoid() {
#if defined(ARDUINO_ARCH_AVR)
  relay.attach(SOLENOID_PIN);
#else
  ledcSetup(SOLENOID_PWM_CHANNEL, SOLENOID_PWM_HZ, SOLENOID_PWM_BITS);
  ledcAttachPin(SOLENOID_PIN, SOLENOID_PWM_CHANNEL);
#endif
}

void writeSolenoidMicroseconds(unsigned int pulse_us) {
#if defined(ARDUINO_ARCH_AVR)
  relay.writeMicroseconds(pulse_us);
#else
  const unsigned long period_us = 1000000UL / SOLENOID_PWM_HZ;
  const unsigned long max_duty = (1UL << SOLENOID_PWM_BITS) - 1UL;
  const unsigned long duty =
    ((unsigned long)pulse_us * max_duty) / period_us;
  ledcWrite(SOLENOID_PWM_CHANNEL, duty);
#endif
}

bool beginLightSensor() {
  const byte addresses[] = {0x23, 0x5C};
  for (byte i = 0; i < sizeof(addresses); i++) {
    const byte address = addresses[i];
    Wire.beginTransmission(address);
    if (Wire.endTransmission() != 0) {
      continue;
    }
    if (lightSensor.begin(BH1750::CONTINUOUS_HIGH_RES_MODE, address, &Wire)) {
      light_sensor_address = address;
      return true;
    }
  }
  light_sensor_address = 0;
  return false;
}


void setup() {

  Serial.begin(9600);
  attachSolenoid();
  // I2C Bus
  Wire.begin();

  light_sensor_ready = beginLightSensor();
  last_sensor_retry_ms = millis();

  writeSolenoidMicroseconds(1000);
}

void loop() {
  handleSerial();

  now = millis();
  if (
    !light_sensor_ready &&
    (unsigned long)(now - last_sensor_retry_ms) >= 1000UL
  ) {
    light_sensor_ready = beginLightSensor();
    last_sensor_retry_ms = now;
  }
  if (light_sensor_ready) {
    lux = lightSensor.readLightLevel();
    if (!isfinite(lux) || lux < 0.0) {
      light_sensor_ready = false;
      last_sensor_retry_ms = now;
    }
  }
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
    writeSolenoidMicroseconds(1000);
    Serial.println("ok stop");
  } else if (strcmp(cmd, "auto on") == 0) {
    auto_run_enabled = true;
    low_lux_since_ms = 0;
    Serial.println("ok auto on");
  } else if (strcmp(cmd, "auto off") == 0) {
    auto_run_enabled = false;
    low_lux_since_ms = 0;
    writeSolenoidMicroseconds(1000);
    Serial.println("ok auto off");
  } else if (strcmp(cmd, "status") == 0) {
    if (light_sensor_ready) {
      lux = lightSensor.readLightLevel();
    }
    Serial.print("ok lux=");
    Serial.print(lux);
    Serial.print(" sensor=");
    Serial.print(light_sensor_ready ? "ok" : "error");
    Serial.print(" addr=");
    if (light_sensor_address) {
      Serial.print("0x");
      Serial.print(light_sensor_address, HEX);
    } else {
      Serial.print("none");
    }
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

  // BH1750 uses negative values for communication/configuration failures.
  // Never interpret a sensor error as a dark marble covering the sensor.
  if (!light_sensor_ready || !isfinite(lux) || lux < 0.0) {
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
  writeSolenoidMicroseconds(2000);
  delay(SOLENOID_ON_MS);
  writeSolenoidMicroseconds(1000);
}

void reboot() {
#if defined(ARDUINO_ARCH_AVR)
  wdt_disable();
  wdt_enable(WDTO_15MS);
  while (1) {}
#else
  ESP.restart();
#endif
}
