
# Ideas comerciales
* SONY: SSE-TN1W
* Zepp Tennis 2

# Resumen Papers

## Papers Tenis de Mesa
1. [Accurate Recognition of Player Identity and Stroke Performance in Table Tennis Using a Smart Wristband](./SmartBandTableTennis.pdf)
2. [Action Recognition and Application of Table Tennis Training Based on IOT Perception](./IoTTableTennis.pdf)

### [Paper 1](./SmartBandTableTennis.pdf)

## Papers Tenis
1. [An Approach to 3D Gyro Sensor Based Motion Analysis in Tennis Forehand Stroke](#resumen-paper-1)
2. [An_embedded_6-axis_sensor_based_recognition_for_tennis_stroke](./An_embedded_6-axis_sensor_based_recognition_for_tennis_stroke.pdf)
3. [Tennis Stroke Classification: ComparingWrist and Racket as IMU Sensor Position](./ComparingWrist&Racket.pdf)

### [Resumen Paper 1](./ForehandGyro.pdf)
### [Resumen Paper 2](./An_embedded_6-axis_sensor_based_recognition_for_tennis_stroke.pdf)
![Paper3modelo](Paper3modelo.png)
Microcontrolador de 6 eixes JY-61. Modulo de control STM32F405 por BLuettoth. Instalado en raqueta, realtime a mobil. (98% golpes, 96% efectos)
3 etapas:
* Reconocemento de golpe (fluctuación de aceleración en ventanas).
* Clasificación de golpe (forehand, backhand, saque) por aceleración e velocidad angular.
* TopSpin o BackSpin rotación de pelota por velocidad angular.

Comparativa de aceleración maxima, desviación estandar e diferencia entre maximo e minimo. O mellor foi desviación estandar. 

Smash e saque valores importantes en Y. Para diferenciar saque de smash, para o saque vai existir unha gran difrencia temporal entre o saque e o ultimo golpe ao contrario que no smash.

### [Resumen Paper 3](./ComparingWrist&Racket.pdf)
**MIRAR REFERENCIAS INTERESANTES**  
Compara muñeca con raqueta. Para clasificar os tipos de golpes usa 4 modelos de machine learning.
Métodos para detectar golpes:
* Acelaración pico con threshold: 9g en x axis
* Calculated accelaration de 3 o 8 gs
* Falsos negativos fora con gaps de 1.25 segundos ou or alternatively gyroscope resultant values within 0.06 s on both sides of the acceleration maxima exceeding ±400 deg.

Machine Learning usados Naive Bayes , support vector machine (SVM), decision and classification tree(CT) random forest (RF) k-nearest neighbor (KNN), neural network (NN) and logistic regression(LG)  Further models include AdaBoost and discriminant analysis.

Usan acceleration and angular velocity  

Mellores resultados SVM linear con datos brutos de acelaración e giroscopio pasada por un PCA.

## Outros
1. [Design and Performance Evaluation of An Arduino Based Activity Tracker](./arduinoactivitytracker.pdf)

## Ideas Traballo relacionado / Referencias


1. L. Chen, J. Hoey, C. D. Nugent, D. J. Cook and Z. Yu, "Sensor-Based
Activity Recognition," in IEEE Transactions on Systems, Man, and
Cybernetics, Part C (Applications and Reviews), vol. 42, no. 6, pp. 790-
808, Nov. 2012.
2.  K. F. Li, A. M. Sevcenco and K. Takano, "Real-Time Classification of
Sports Movement Using Adaptive Clustering," Complex, Intelligent and
Software Intensive Systems (CISIS), 2012 Sixth International
Conference on, Palermo, 2012, pp. 68-75.
3.  D. Conaghan, P. Kelly, and N. E. O’Connor. Game, shot and match:
Event-based indexing of tennis. In 9th International Workshop on
Content-Based Multimedia Indexing, pages 96–102, 2011.
4. D. Connaghan, S. Hughes, G. May, K. O’Brien, P. Kelly, C. ´O Conaire,
N. E. O’Connor, D. O’Gorman, G. Warrington, A. F. Smeaton, and N.
Moyna. A sensing platform for physiological and contextual feedback to
tennis athletes. In Body Sensor Networks (BSN) Workshop, pages 224–
229, 2009.
5. D. Connaghan, P. Kelly, et al., “Multi-sensor classification of tennis
strokes,” IEEE Sensors, pp. 1437–1440, 2011.
6. R. Srivastava, A. Patwari, et al., “Efficient Characterization of Tennis
Shots and Game Analysis using Wearable Sensors Data”, IEEE Sensors,
pp. 1–4, 2015.
7. http://www.st.com/content/ccc/resource/technical/document/datasheet/ef/92/76/6d/bb/c2/4f/f7/DM00037051.pdf/files/DM00037051.pdf/jcr:content/translations/en.DM00037051.pdf
# Borrador Anteproxecto
[Borrador](./MUEI_Solicitude_Anteproxecto.docx)
