////
const int DIR1 = 14;
const int STEP1 = 12;
//
const int DIR2 = 15;
const int STEP2 = 13;
////
const int DIR3 = 4;
const int STEP3 = 5;
//
int counter = 0;
////
////
void setup()
{
  Serial.begin(9600);
  
  pinMode(STEP1, OUTPUT);
  pinMode(DIR1, OUTPUT);
  
  pinMode(STEP2, OUTPUT);
  pinMode(DIR2, OUTPUT);
//
//  pinMode(STEP3, OUTPUT);
//  pinMode(DIR3, OUTPUT);

  
  digitalWrite(DIR1, LOW);
  digitalWrite(DIR2, LOW);
//  digitalWrite(DIR3, LOW);

  
  tone(STEP1, 100);
  tone(STEP2, 100);
//  tone(STEP3, 100);
  
  Serial.println("test1");
  delay(500);
}
////
void loop()
{
  
  counter ++;
  Serial.println("boardTest");


     


}
//
//
