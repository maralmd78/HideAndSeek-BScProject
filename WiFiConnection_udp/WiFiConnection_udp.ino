#include <ESP8266WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>

// Set WiFi credentials
#define WIFI_SSID "IRLab"
#define WIFI_PASS "Khosravi503"
#define UDP_PORT 80


// UDP
WiFiUDP UDP;
char packet[255];

//// indicate the max number of bytes in json message
StaticJsonDocument<500> doc;
DeserializationError error;

/////
const int DIR1 = 14;
const int STEP1 = 12;
int speed1 = 0;
bool direction1 = false;
////
const int DIR2 = 15;
const int STEP2 = 13;
int speed2 = 0;
bool direction2 = false;
////
const int DIR3 = 4;
const int STEP3 = 5;
int speed3 = 0;
bool direction3 = false;
////

void motorSetup(int S1, int S2, int S3, int D1, int D2, int D3)
{
  
//  digitalWrite(DIR1, (D1==1 ? HIGH : LOW));
  if(D1==1){
    digitalWrite(DIR1, HIGH);
  }
  else if(D1==0)
  {
    digitalWrite(DIR1, LOW);
  }
  digitalWrite(DIR2, (D2==1 ? HIGH : LOW));
  digitalWrite(DIR3, (D3==1 ? HIGH : LOW));
  tone(STEP1, S1);
  tone(STEP2, S2);
  tone(STEP3, S3);

}


void setup() {
  Serial.begin(9600);
  Serial.println();

  pinMode(STEP1, OUTPUT);
  pinMode(DIR1, OUTPUT);
  pinMode(STEP2, OUTPUT);
  pinMode(DIR2, OUTPUT);
  pinMode(STEP3, OUTPUT);
  pinMode(DIR3, OUTPUT);
  
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  // Connecting to WiFi...
  Serial.print("Connecting to ");
  Serial.print(WIFI_SSID);

  // Loop continuously while WiFi is not connected
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(100);
    Serial.print(".");
  }

  // Connected to WiFi
  Serial.println();
  Serial.print("Successfully connected! IP address: ");
  Serial.println(WiFi.localIP());

  // Begin listening to UDP port
  UDP.begin(UDP_PORT);
  Serial.print("Listening on UDP port ");
  Serial.println(UDP_PORT);
}

void loop() {
  int packetSize = UDP.parsePacket();
  if(packetSize)
  {
    Serial.print("Received packet! Size: ");
    Serial.println(packetSize);
    int len = UDP.read(packet, 255);
    if (len > 0)
    {
      packet[len] = '\0';
    }
    
    Serial.print("Packet: ");
    Serial.println(packet);
    error = deserializeJson(doc, packet);
    if(!error)
        {
          speed1 = doc["S1"].as<int>();
          speed2 = doc["S2"].as<int>();
          speed3 = doc["S3"].as<int>();
          direction1 = doc["D1"].as<int>();
          direction2 = doc["D2"].as<int>();
          direction3 = doc["D3"].as<int>();
          
//          Serial.println(speed1);
//          Serial.println(speed2);
//          Serial.println(speed3);
//          Serial.println(direction1);
//          Serial.println(direction2);
//          Serial.println(direction3);
          
          motorSetup(speed1, speed2, speed3, direction1, direction2, direction3);
          
        }
    
    UDP.endPacket();
  }
  else
  {
    noTone(STEP1);
    noTone(STEP2);
    noTone(STEP3);
  }

  }
  
