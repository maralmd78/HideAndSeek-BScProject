#include <ESP8266WiFi.h>
#include <ArduinoJson.h>

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


const char* ssid = "IRLab";
const char* password = "Khosravi503";
WiFiServer server(80);


//// indicate the max number of bytes in json message
StaticJsonDocument<150> doc;
DeserializationError error;


void motorSetup(int S1, int S2, int S3, bool D1, bool D2, bool D3)
{
  
  digitalWrite(DIR1, (D1 ? HIGH : LOW));
  digitalWrite(DIR2, (D2 ? HIGH : LOW));
  digitalWrite(DIR3, (D3 ? HIGH : LOW));
  tone(STEP1, S1);
  tone(STEP2, S2);
  tone(STEP3, S3);

}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  
  pinMode(STEP1, OUTPUT);
  pinMode(DIR1, OUTPUT);
  
  pinMode(STEP2, OUTPUT);
  pinMode(DIR2, OUTPUT);

  pinMode(STEP3, OUTPUT);
  pinMode(DIR3, OUTPUT);
  
  
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected.");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());

  server.begin();

}

void loop() {
  WiFiClient client = server.available();
  
  if (client) {
    Serial.println("New client connected.");
    
    while (client.connected()) {
      if (client.available()) {
        String request = client.readStringUntil('\r');
        request.trim();
        Serial.println(request);
        error = deserializeJson(doc, request);
        Serial.println(error.c_str());
        if(!error)
        {
          speed1 = doc["S1"].as<int>();
          speed2 = doc["S2"].as<int>();
          speed3 = doc["S3"].as<int>();
          direction1 = doc["D1"].as<bool>();
          direction2 = doc["D2"].as<bool>();
          direction3 = doc["D3"].as<bool>();
          
          Serial.println(speed1);
          Serial.println(speed2);
          Serial.println(speed3);
          Serial.println(direction1);
          Serial.println(direction2);
          Serial.println(direction3);
          motorSetup(speed1, speed2, speed3, direction1, direction2, direction3);
        }
        
        client.println("OK");
        
      }
    }
    
    motorSetup(0, 0, 0, 0, 0, 0);
    client.stop();
    Serial.println("Client disconnected.");
  }
}
