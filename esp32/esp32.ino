#include <SoftwareSerial.h>
#include <Adafruit_Fingerprint.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// Wiring: R307 TX -> D5 (GPIO14), RX -> D6 (GPIO12)
SoftwareSerial mySerial(14, 12); 
Adafruit_Fingerprint finger = Adafruit_Fingerprint(&mySerial);

// LCD Setup: Address 0x27 is common, SDA -> D2 (GPIO4), SCL -> D1 (GPIO5)
LiquidCrystal_I2C lcd(0x27, 16, 2);

// Base64 lookup table
const char b64_table[] PROGMEM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

uint8_t b64Buffer[3];
int b64Idx = 0;

void setup() {
  // Serial to PC
  Serial.begin(115200);
  
  // Initialize LCD
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("System Starting");
  
  delay(1000);
  Serial.println(F("\n[SYSTEM] ESP8266 Online"));
  
  // R307 sensor
  finger.begin(57600); 
  
  if (finger.verifyPassword()) {
    Serial.println(F("[SYSTEM] Sensor Linked"));
    lcd.clear();
    lcd.print("Sensor Ready");
  } else {
    Serial.println(F("[CRITICAL] Sensor connection failed"));
    lcd.clear();
    lcd.print("Sensor Error!");
    while (1) yield();
  }
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == F("READY")) {
      handleImageCapture();
    } 
    else if (cmd.startsWith(F("RESULT:"))) {
      handleResult(cmd);
    }
  }
  yield();
}

// Optimized Base64 streamer
void encodeAndPrint(uint8_t* chunk, int len) {
  uint32_t val = 0;
  if (len == 3) {
    val = (uint32_t)chunk[0] << 16 | (uint32_t)chunk[1] << 8 | (uint32_t)chunk[2];
    Serial.print((char)pgm_read_byte(&b64_table[(val >> 18) & 0x3F]));
    Serial.print((char)pgm_read_byte(&b64_table[(val >> 12) & 0x3F]));
    Serial.print((char)pgm_read_byte(&b64_table[(val >> 6) & 0x3F]));
    Serial.print((char)pgm_read_byte(&b64_table[val & 0x3F]));
  } else if (len == 2) {
    val = (uint32_t)chunk[0] << 8 | (uint32_t)chunk[1];
    Serial.print((char)pgm_read_byte(&b64_table[(val >> 10) & 0x3F]));
    Serial.print((char)pgm_read_byte(&b64_table[(val >> 4) & 0x3F]));
    Serial.print((char)pgm_read_byte(&b64_table[(val << 2) & 0x3F]));
    Serial.print('=');
  } else if (len == 1) {
    val = (uint32_t)chunk[0];
    Serial.print((char)pgm_read_byte(&b64_table[(val >> 2) & 0x3F]));
    Serial.print((char)pgm_read_byte(&b64_table[(val << 4) & 0x3F]));
    Serial.print("==");
  }
}

void streamByte(uint8_t b) {
  b64Buffer[b64Idx++] = b;
  if (b64Idx == 3) {
    encodeAndPrint(b64Buffer, 3);
    b64Idx = 0;
  }
}

void handleImageCapture() {
  Serial.println(F("[ACTION] Waiting for finger..."));
  lcd.clear();
  lcd.print("Place Finger");
  
  uint8_t p = -1;
  while (p != FINGERPRINT_OK) {
    p = finger.getImage();
    if (p == FINGERPRINT_NOFINGER) { yield(); } 
    else if (p != FINGERPRINT_OK) {
      Serial.println(F("[ERROR] Try again"));
      lcd.setCursor(0, 1);
      lcd.print("Capture Error");
      return;
    }
  }

  lcd.clear();
  lcd.print("Capturing...");

  // Request high-res image transfer from sensor
  uint8_t upImageCmd[] = {0xEF, 0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0x00, 0x03, 0x0A, 0x00, 0x0E};
  mySerial.write(upImageCmd, 12);

  // Wait for sensor ACK (12 bytes)
  unsigned long timeout = millis();
  while (mySerial.available() < 12 && (millis() - timeout < 1000)) yield();
  for (int i = 0; i < 12; i++) { if (mySerial.available()) mySerial.read(); }

  Serial.println(F("IMG_START"));
  b64Idx = 0;

  // 1. BMP Header (54 bytes) - Exactly as expected for 256x144 8-bit
  uint8_t bmpHeader[54] = {
    0x42, 0x4D, 0x36, 0x90, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x36, 0x04, 0x00, 0x00, 
    0x28, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x90, 0x00, 0x00, 0x00, 0x01, 0x00, 
    0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x90, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
  };
  for(int i=0; i<54; i++) streamByte(bmpHeader[i]);
  
  // 2. Grayscale Palette (1024 bytes)
  for(int i=0; i<256; i++) {
    streamByte(i); streamByte(i); streamByte(i); streamByte(0);
  }

  uint32_t totalPixelsProcessed = 0;
  uint8_t packet[256];
  const uint32_t requiredPixels = 36864;

  // 3. Image Data Loop
  while (totalPixelsProcessed < requiredPixels) {
    yield();
    
    // Look for Header 0xEF01
    bool headerFound = false;
    unsigned long searchStart = millis();
    while(!headerFound && (millis() - searchStart < 2500)) {
       if(mySerial.available()) {
         if(mySerial.read() == 0xEF) {
           while(!mySerial.available() && (millis() - searchStart < 2500)) yield();
           if(mySerial.read() == 0x01) headerFound = true;
         }
       }
       yield();
    }
    
    // If header is not found, break and proceed to zero padding
    if(!headerFound) break;

    // Read Packet info (Address 4 bytes, ID 1 byte, Len 2 bytes)
    for(int i=0; i<7; i++) {
      while(!mySerial.available()) yield();
      packet[i] = mySerial.read();
    }
    
    uint16_t len = (uint16_t)packet[5] << 8 | packet[6];
    int dataLen = len - 2;

    // Read pixels from this packet
    for(int i=0; i<dataLen; i++) {
      while(!mySerial.available()) yield();
      uint8_t b = mySerial.read();
      
      if (totalPixelsProcessed < requiredPixels) {
        streamByte(b);
        totalPixelsProcessed++;
      }
    }

    // Read checksum (2 bytes)
    for(int i=0; i<2; i++) {
      while(!mySerial.available()) yield();
      mySerial.read();
    }
    
    // Visual progress on LCD
    if(totalPixelsProcessed % 4096 == 0) {
       lcd.setCursor(0, 1);
       lcd.print("Sending: ");
       lcd.print((totalPixelsProcessed * 100) / requiredPixels);
       lcd.print("% ");
    }

    Serial.println(); 
    delay(1);
  }

  // 4. Zero Padding - Fulfill the required size if sensor data was short
  while (totalPixelsProcessed < requiredPixels) {
    streamByte(0x00);
    totalPixelsProcessed++;
  }

  // Finalize Base64 padding
  if (b64Idx > 0) encodeAndPrint(b64Buffer, b64Idx);

  Serial.println();
  Serial.println(F("IMG_END"));
  
  lcd.clear();
  lcd.print("Processing...");
  
  // Debug output to verify size
  Serial.print(F("[DEBUG] Bytes: "));
  Serial.println(totalPixelsProcessed);
}

void handleResult(String cmd) {
  // Expected format: RESULT:AB-:41
  int first = cmd.indexOf(':');
  int second = cmd.indexOf(':', first + 1);
  
  
  if (first != -1 && second != -1) {
    String bg = cmd.substring(first + 1, second);
    String conf = cmd.substring(second + 1);
    
    lcd.clear();
    lcd.print("Blood Group:");
    lcd.setCursor(0, 1);
    lcd.print(bg);
    lcd.print(" (");
    lcd.print(conf);
    lcd.print("%)");
    delay(5000);

    Serial.println(F("ACK")); 
  }
}