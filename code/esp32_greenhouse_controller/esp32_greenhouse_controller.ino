/*
 * esp32_greenhouse_controller.ino
 * ================================
 * Phase 6: Real hardware deployment of the trained PPO greenhouse controller.
 *
 * What this does:
 *   Every 10 minutes (matching the simulation timestep):
 *     1. Reads 5 sensors via GPIO pins
 *     2. Normalises readings to [0,1] using real data ranges from parameter_ranges.json
 *     3. Runs the neural network forward pass using weights from nn_weights.h
 *     4. Picks the best action (argmax of output layer)
 *     5. Fires the corresponding relay for 10 minutes
 *     6. Logs all readings and decisions to Serial (for monitoring)
 *
 * Hardware required:
 *   ESP32 development board (any variant with at least 5 GPIO output pins)
 *   DHT22 sensor              (temperature + humidity)
 *   Capacitive soil moisture sensor (soil moisture)
 *   LDR (photoresistor)       (light intensity proxy)
 *   MQ135 gas sensor          (CO2 and VOC proxy)
 *   5-channel relay board     (controls actuators)
 *   Actuators: ventilation fan, water pump, grow light strip,
 *              small heater, CO2 valve (optional)
 *
 * Wiring:
 *   DHT22 data     → GPIO 4
 *   Soil sensor    → GPIO 34 (ADC1, 12-bit)
 *   LDR            → GPIO 35 (ADC1, 12-bit)
 *   MQ135          → GPIO 32 (ADC1, 12-bit)
 *   Relay 1 (vent) → GPIO 16
 *   Relay 2 (pump) → GPIO 17
 *   Relay 3 (lights)→ GPIO 18
 *   Relay 4 (heater)→ GPIO 19
 *   Relay 5 (CO2)  → GPIO 21
 *
 * Libraries needed (install via Arduino Library Manager):
 *   DHT sensor library by Adafruit
 *   Adafruit Unified Sensor
 *
 * Before flashing:
 *   Run convert_weights.py in Colab to generate nn_weights.h
 *   Place nn_weights.h in the same folder as this .ino file
 */

#include <DHT.h>
#include <math.h>
#include "nn_weights.h"   // Neural network weights from Phase 5 training

// ── Pin definitions ──────────────────────────────────────────────────────────
#define DHT_PIN        4
#define DHT_TYPE       DHT22
#define SOIL_PIN       34   // Analog input
#define LDR_PIN        35   // Analog input
#define MQ135_PIN      32   // Analog input

// Relay pins — HIGH = relay ON = actuator active
#define RELAY_VENT     16
#define RELAY_PUMP     17
#define RELAY_LIGHTS   18
#define RELAY_HEATER   19
#define RELAY_CO2      21

// ── Simulation parameters (from parameter_ranges.json) ───────────────────────
// These MUST match the ranges used during training.
// Order: [temperature, humidity, light, co2, soil, voc, time_of_day]
const float PARAM_MIN[7] = {  5.0f,   20.0f,      0.0f,  400.0f,  10.0f,    0.0f,  0.0f };
const float PARAM_MAX[7] = { 45.0f,   85.0f,  35000.0f, 2000.0f,  90.0f, 2000.0f, 23.0f };

// ── Action definitions ────────────────────────────────────────────────────────
// Must match the order in greenhouse_env.py ACTIONS dict
#define ACTION_DO_NOTHING   0
#define ACTION_OPEN_VENTS   1
#define ACTION_WATER_PLANTS 2
#define ACTION_GROW_LIGHTS  3
#define ACTION_HEATER       4
#define ACTION_INJECT_CO2   5

const char* ACTION_NAMES[] = {
    "Do nothing",
    "Open vents",
    "Water plants",
    "Grow lights on",
    "Heater on",
    "Inject CO2"
};

// ── Control interval ─────────────────────────────────────────────────────────
// 10 minutes = 600,000 ms
// This matches the simulation timestep (10-minute intervals in the dataset)
#define CONTROL_INTERVAL_MS 600000UL

// ── Global objects ────────────────────────────────────────────────────────────
DHT dht(DHT_PIN, DHT_TYPE);
unsigned long last_control_time = 0;
int step_count = 0;

// ── Sensor calibration (adjust for your specific sensors) ────────────────────
// Soil moisture: ADC value when fully dry vs fully wet
// Measure these values with your sensor before deploying
#define SOIL_DRY_ADC   3500   // ADC reading in completely dry soil
#define SOIL_WET_ADC    800   // ADC reading in water-saturated soil

// LDR to lux: rough linear mapping, calibrate against a reference light meter
#define LDR_DARK_ADC      0   // ADC in darkness
#define LDR_BRIGHT_ADC 4095   // ADC in bright sunlight

// MQ135 to CO2: rough linear mapping, calibrate after 24h warm-up in fresh air
#define MQ135_CLEAN_ADC  400   // ADC in clean air (~400 ppm)
#define MQ135_HIGH_ADC  2500   // ADC in high CO2 (~2000 ppm)


// ── Neural network forward pass ───────────────────────────────────────────────
/*
 * Implements the same computation as the Python PPO policy network.
 *
 * h1     = tanh(W1 * obs + b1)       [7 inputs  → 64 hidden units]
 * h2     = tanh(W2 * h1  + b2)       [64 hidden → 64 hidden units]
 * logits = W3 * h2 + b3              [64 hidden → 6 action scores]
 * action = argmax(logits)
 *
 * No softmax needed: we only want the highest-scoring action.
 *
 * Memory: h1[64] + h2[64] = 128 floats = 512 bytes stack usage
 * Time:   7*64 + 64*64 + 64*6 = 4,672 multiply-add operations
 * On ESP32 at 240MHz: well under 1ms execution time
 */
int run_policy(float obs[7]) {
    float h1[64];
    float h2[64];
    float logits[6];

    // Layer 1: h1 = tanh(W1 * obs + b1)
    for (int i = 0; i < 64; i++) {
        float sum = b1[i];
        for (int j = 0; j < 7; j++) {
            sum += W1[i][j] * obs[j];
        }
        h1[i] = tanhf(sum);
    }

    // Layer 2: h2 = tanh(W2 * h1 + b2)
    for (int i = 0; i < 64; i++) {
        float sum = b2[i];
        for (int j = 0; j < 64; j++) {
            sum += W2[i][j] * h1[j];
        }
        h2[i] = tanhf(sum);
    }

    // Layer 3 (output): logits = W3 * h2 + b3
    int best_action = 0;
    float best_logit = -1e9f;

    for (int i = 0; i < 6; i++) {
        float sum = b3[i];
        for (int j = 0; j < 64; j++) {
            sum += W3[i][j] * h2[j];
        }
        logits[i] = sum;
        if (sum > best_logit) {
            best_logit = sum;
            best_action = i;
        }
    }

    return best_action;
}

// ── Normalise a sensor reading to [0, 1] ─────────────────────────────────────
float normalise(float value, int param_idx) {
    float norm = (value - PARAM_MIN[param_idx]) / (PARAM_MAX[param_idx] - PARAM_MIN[param_idx]);
    if (norm < 0.0f) norm = 0.0f;
    if (norm > 1.0f) norm = 1.0f;
    return norm;
}

// ── Read all sensors and return observation array ────────────────────────────
bool read_sensors(float obs[7]) {

    // Temperature and humidity from DHT22
    float temp = dht.readTemperature();
    float hum  = dht.readHumidity();

    if (isnan(temp) || isnan(hum)) {
        Serial.println("[ERROR] DHT22 read failed. Retrying in 2s...");
        delay(2000);
        temp = dht.readTemperature();
        hum  = dht.readHumidity();
        if (isnan(temp) || isnan(hum)) {
            Serial.println("[ERROR] DHT22 failed again. Skipping this step.");
            return false;
        }
    }

    // Soil moisture from capacitive sensor (inverted: dry = high ADC)
    int soil_adc = analogRead(SOIL_PIN);
    float soil   = map(soil_adc, SOIL_DRY_ADC, SOIL_WET_ADC, 10, 90);
    soil = constrain(soil, 10.0f, 90.0f);

    // Light from LDR (linear mapping to lux estimate)
    int ldr_adc = analogRead(LDR_PIN);
    float lux   = map(ldr_adc, LDR_DARK_ADC, LDR_BRIGHT_ADC, 0, 35000);
    lux = constrain(lux, 0.0f, 35000.0f);

    // CO2 proxy from MQ135
    int co2_adc = analogRead(MQ135_PIN);
    float co2   = map(co2_adc, MQ135_CLEAN_ADC, MQ135_HIGH_ADC, 400, 2000);
    co2 = constrain(co2, 400.0f, 2000.0f);

    // VOC: placeholder until a dedicated VOC sensor (e.g. SGP30) is added
    // For now, estimate from MQ135 (rough proxy only)
    float voc = (co2 - 400.0f) * 0.5f;
    voc = constrain(voc, 0.0f, 2000.0f);

    // Time of day in hours (0-23)
    // Replace with DS3231 RTC reading for accurate time
    unsigned long ms = millis();
    float tod = fmod((float)(ms / 3600000.0f), 24.0f);

    // Build observation array matching greenhouse_env.py PARAM_NAMES order:
    // [temperature_c, humidity_pct, light_lux, co2_ppm, soil_moisture_pct, voc_ppb, time_of_day]
    obs[0] = normalise(temp, 0);
    obs[1] = normalise(hum,  1);
    obs[2] = normalise(lux,  2);
    obs[3] = normalise(co2,  3);
    obs[4] = normalise(soil, 4);
    obs[5] = normalise(voc,  5);
    obs[6] = normalise(tod,  6);

    // Print raw readings to Serial for monitoring
    Serial.printf("  Temp:     %.1f°C\n",  temp);
    Serial.printf("  Humidity: %.1f%%\n",  hum);
    Serial.printf("  Light:    %.0f lux\n",lux);
    Serial.printf("  CO2:      %.0f ppm\n",co2);
    Serial.printf("  Soil:     %.1f%%\n",  soil);
    Serial.printf("  VOC:      %.0f ppb\n",voc);
    Serial.printf("  Time:     %.1fh\n",   tod);

    return true;
}

// ── Execute action by firing correct relay ────────────────────────────────────
void execute_action(int action) {
    // Turn all relays off first (one action at a time)
    digitalWrite(RELAY_VENT,   LOW);
    digitalWrite(RELAY_PUMP,   LOW);
    digitalWrite(RELAY_LIGHTS, LOW);
    digitalWrite(RELAY_HEATER, LOW);
    digitalWrite(RELAY_CO2,    LOW);

    switch (action) {
        case ACTION_DO_NOTHING:
            // No relay fired
            break;
        case ACTION_OPEN_VENTS:
            digitalWrite(RELAY_VENT, HIGH);
            break;
        case ACTION_WATER_PLANTS:
            digitalWrite(RELAY_PUMP, HIGH);
            break;
        case ACTION_GROW_LIGHTS:
            digitalWrite(RELAY_LIGHTS, HIGH);
            break;
        case ACTION_HEATER:
            digitalWrite(RELAY_HEATER, HIGH);
            break;
        case ACTION_INJECT_CO2:
            digitalWrite(RELAY_CO2, HIGH);
            break;
        default:
            Serial.printf("[WARN] Unknown action: %d\n", action);
            break;
    }
}


// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("========================================");
    Serial.println("Greenhouse AI Controller v1.0");
    Serial.println("Phase 6: PPO Policy Deployment");
    Serial.println("========================================");
    Serial.println();

    // Initialise DHT sensor
    dht.begin();

    // Set all relay pins as output, start LOW (relays off)
    int relay_pins[] = {RELAY_VENT, RELAY_PUMP, RELAY_LIGHTS, RELAY_HEATER, RELAY_CO2};
    for (int i = 0; i < 5; i++) {
        pinMode(relay_pins[i], OUTPUT);
        digitalWrite(relay_pins[i], LOW);
    }

    // Warm up MQ135 (needs 24-48 hours for accurate readings in a new sensor)
    Serial.println("[INFO] Sensors initialised.");
    Serial.println("[INFO] Note: MQ135 requires 24h warm-up for accurate CO2 readings.");
    Serial.println("[INFO] First control decision in 10 seconds...");
    Serial.println();

    // Short initial delay for sensor stabilisation
    delay(10000);

    Serial.println("[READY] Controller active. One decision every 10 minutes.");
    Serial.println();
}


// ── Main loop ─────────────────────────────────────────────────────────────────
void loop() {
    unsigned long now = millis();

    // Only act every CONTROL_INTERVAL_MS (10 minutes)
    if (now - last_control_time >= CONTROL_INTERVAL_MS || last_control_time == 0) {
        last_control_time = now;
        step_count++;

        Serial.printf("--- Step %d | Time: %lu min ---\n",
                      step_count, now / 60000UL);

        // 1. Read sensors
        float obs[7];
        bool sensors_ok = read_sensors(obs);

        if (!sensors_ok) {
            Serial.println("[SKIP] Sensor read failed. Waiting for next interval.");
            Serial.println();
            return;
        }

        // 2. Run neural network inference
        int action = run_policy(obs);

        // 3. Execute action
        execute_action(action);

        // 4. Log decision
        Serial.printf("  Decision: [%d] %s\n", action, ACTION_NAMES[action]);
        Serial.println();
    }

    // Small delay to prevent watchdog timeout
    delay(100);
}
