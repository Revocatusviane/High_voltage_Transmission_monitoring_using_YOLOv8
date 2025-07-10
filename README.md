The directory has got files which was used to complete a project.
The project was specifically for High-voltage-transmission-lines protection.
it was implemented using flask framework, it intergrated YOLOv8,Arduino language(C++),AND DecisionTreeClassifier.
The YOLOv8 was for computer vision(for Protection insulation checking and decisionining.
C++ was specifically for ESP32 as , the esp32 had to read data (CURENT AND VOLTAGE)....
FROM the external circuit and feed the decicionTreeClasssifier.
The decicionTreeClasssifier was impleted as a model to classify (OVER VOLTAGES, UNDER VOLTAGES, OVERCURRENT).
The project used IOT based principle to easen the communication via a wifi between the flask APP and the esp .
The app is not hosted, testing was done locally.
