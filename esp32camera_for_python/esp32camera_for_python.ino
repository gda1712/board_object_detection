#include <WiFi.h>
#include <esp32cam.h>
#include <WebServer.h>
 
const char* WIFI_SSID = "**********";
const char* WIFI_PASS = "**********";
WebServer server(80);
 
static auto hiRes = esp32cam::Resolution::find(1000, 1000);
void serveJpg()
{
  // Delay for stabilization
  delay(50);
  auto frame = esp32cam::capture();
  if (frame == nullptr) {
    Serial.println("CAPTURE FAIL");
    server.send(503, "", "");
    return;
  }
  Serial.printf("CAPTURE OK %dx%d %db\n", frame->getWidth(), frame->getHeight(),
                static_cast<int>(frame->size()));

  server.setContentLength(frame->size());
  server.send(200, "image/jpeg");
  WiFiClient client = server.client();
  frame->writeTo(client);
}

void handleJpgHi()
{
  // Set high resolution
  if (!esp32cam::Camera.changeResolution(hiRes)) {
    Serial.println("SET-HI-RES FAIL");
  }
  serveJpg();
}
 
void  setup(){
  pinMode(4, OUTPUT);
  Serial.begin(115200);
  Serial.println();
  {
    using namespace esp32cam;
    Config cfg;
    cfg.setPins(pins::AiThinker);
    cfg.setResolution(hiRes);
    cfg.setBufferCount(6);  // Increased for better stability
    cfg.setJpeg(85);
 
    bool ok = Camera.begin(cfg);
    Serial.println(ok ? " CAMERA OK" : "CAMERA FAIL");
  }
  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
  Serial.print("http://");
  Serial.println(WiFi.localIP());
  Serial.println("  /cam-hi.jpg");

  server.on("/cam-hi.jpg", handleJpgHi);
 
  server.begin();
}
 
void loop()
{
  digitalWrite(4, HIGH);
  server.handleClient();
}