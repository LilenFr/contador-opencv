## instalar e importar opencv ##

# en la terminal: pip install opencv-contrib-python==4.6.0.66

# importar
import cv2
import numpy as np

## cargar y visualizar imagen ##

# cargar y leer la imagen
original=cv2.imread('monedas.jpg')

# pasar imagen a escala de grises
gris=cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)

# mostrar resultados
#cv2.imshow('grises', gris)
#cv2.waitKey(0)

## eliminar ruido ##

# 1) suavizar imagen con desenfoque gaussiano
valorGauss=1 #alterar según corresponda

gauss=cv2.GaussianBlur(gris, (valorGauss,valorGauss), 0)

# mostrar resultados
#cv2.imshow('grises', gris)
#cv2.imshow('gauss', gauss)
#cv2.waitKey(0)

# 2) encontrar los bordes con canny
canny=cv2.Canny(gauss, 60,100)

# mostrar resultados
#cv2.imshow('grises', gris)
#cv2.imshow('gauss', gauss)
#cv2.imshow('canny', canny)
#cv2.waitKey(0)

# Luego de encontrar los contornos hay que determinar el contorno que se quiere capturar. En este caso es el contorno mayor, es decir, la forma circular de las monedas, que es lo que necesitamos para poder contarlas. Es pertinente entonces eliminar el ruido interno (los detalles visuales dentro de las monedas), ya que no nos sirven y pueden causar problemas. Conceptos referencia: clausura y morfología.

valorKernel=7 #alterar según corresponda

kernel=np.ones((valorKernel,valorKernel),np.uint8)

cierre=cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

contornos, jerarquía=cv2.findContours(cierre.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("monedas encontradas: {}".format(len(contornos)))
cv2.drawContours(original, contornos, -1, (0,0,255),2)

# mostrar resultados
#cv2.imshow('Grises',gris)
#cv2.imshow('gauss',gauss)
#cv2.imshow('canny',canny)
#cv2.imshow('resultado', original)
#cv2.waitKey(0)

cv2.imshow("Resultado", original)
cv2.waitKey(0)