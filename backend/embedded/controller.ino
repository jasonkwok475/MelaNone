#define STEP_PIN 26   // PUL-
#define DIR_PIN  25   // DIR- check if flipped?
#define LIMIT_PIN 27  // Limit switch pin

// speed
// smaller number = faster rotation
#define STEP_DELAY_US 1000
define FWD_MOVE 200

// define directions
#define FORWARD 0
#define BACKWARD 1

int prevComm = 0;
int pos = 0;
int comm = 0;

void setup() {
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(LIMIT_PIN, INPUT);  // config limit switch pin, might need to pullup (and update states below)

  // Idle states
  digitalWrite(STEP_PIN, HIGH);
  digitalWrite(DIR_PIN, FORWARD);  

  Serial.begin(115200);
  delay(500);
  Serial.println("Stepper running...");

  Serial.setTimeout(1);
}

void loop() {
  // waiting for serial to initialize
  while(!Serial.available());

  checkSerial();
  delay(100);
}

void checkSerial(){
  // read inputs via serial
  comm = Serial.readString().toInt();

  if (prevComm != comm) {
    if (comm == 1){
      moveForward();
    }
    else if (comm == 3){
      rst();
    }
    else{
      // do nothing, prev command is either duplicate or response from FW
    }

    prevComm = comm;
  }
}

void moveForward(){
    if pos <= 3{ 
        // move forward
        digitalWrite(DIR_PIN, FORWARD);

        for(int s = 0; s < FWD_MOVE; s++){
            digitalWrite(STEP_PIN, LOW);
            delayMicroseconds(STEP_DELAY_US);
            digitalWrite(STEP_PIN, HIGH);
            delayMicroseconds(STEP_DELAY_US);
        }
    pos += 1;
    Serial.println(2); // print done
    }
    else{
    Serial.println(-1); // print error
    }

}

void rst(){
  // move forward a bit in case we're currently on the switch
  digitalWrite(DIR_PIN, FORWARD);
    for(int s = 0; s < 50; s++){
      digitalWrite(STEP_PIN, LOW);
      delayMicroseconds(STEP_DELAY_US);
      digitalWrite(STEP_PIN, HIGH);
      delayMicroseconds(STEP_DELAY_US);
    }
// move backward until limit switch is triggered
  digitalWrite(DIR_PIN, BACKWARD);
  while(digitalRead(LIMIT_PIN) == HIGH) {
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(STEP_DELAY_US);
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(STEP_DELAY_US);
  }
  
  pos = 0; 
  Serial.println(2); // print done
}