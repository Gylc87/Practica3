import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pylab 
import cv2
from numpy.core.fromnumeric import size 
import skimage
from skimage import io
import math
import PIL
import imutils
from PIL import ImageTk
from PIL import Image as Im
fila = 4
columna = 3

image1 = cv2.imread("img1.jpg")
image2 = cv2.imread("img2.jpg")

img1 = cv2.resize(image1, dsize=(550, 350), interpolation=cv2.INTER_CUBIC)
img2 = cv2.resize(image2, dsize=(550, 350), interpolation=cv2.INTER_CUBIC)

imagen1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
imagen2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

def MostrarImagenes(operacion,imagen1,imagen2,redimgop):
    global fila 
    global columna
    fig = plt.figure(figsize=(10,7), constrained_layout=True)
    fig.add_subplot(fila,columna,1)
    plt.imshow(imagen1)
    plt.axis('off')
    plt.title("Imagen 1")

    fig.add_subplot(fila,columna,4)
    color = ('g','b','r')
    for i, c in enumerate(color):
        hist = cv2.calcHist([imagen1], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])

    plt.title("Histograma img 1")
    fig.add_subplot(fila,columna,7)
    #aqui va el calculo del ecualizado
    img_to_yuv = cv2.cvtColor(imagen1,cv2.COLOR_RGB2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    equaimg1 = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2RGB)
    color = ('g','b','r')
    for i, c in enumerate(color):
        hist = cv2.calcHist([equaimg1], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
        
    plt.title("Histograma img 1 Ecualizada")

    fig.add_subplot(fila,columna,10)
    plt.imshow(equaimg1)
    plt.axis('off')
    plt.title("Imagen 1 Ecualizada")
    #-----------------2da Imagen-------------------------
    fig.add_subplot(fila,columna,3)
    plt.imshow(imagen2)
    plt.axis('off')
    plt.title("Imagen 2")

    fig.add_subplot(fila,columna,6)
    color = ('g','b','r')
    for i, c in enumerate(color):
        hist = cv2.calcHist([imagen2], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])

    plt.title("Histograma img 2")
    fig.add_subplot(fila,columna,9)
    #aqui va el calculo del ecualizado
    img_to_yuv = cv2.cvtColor(imagen2,cv2.COLOR_RGB2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    equaimg2 = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2RGB)
    color = ('g','b','r')
    for i, c in enumerate(color):
        hist = cv2.calcHist([equaimg2], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
        
    plt.title("Histograma img 2 Ecualizada")

    fig.add_subplot(fila,columna,12)
    plt.imshow(equaimg2)
    plt.axis('off')
    plt.title("Imagen 2 Ecualizada")
    #--------------------Operacion------------------
    fig.add_subplot(fila,columna,2)
    plt.imshow(redimgop)
    plt.axis('off')
    plt.title("Imagen 2:"+operacion)

    fig.add_subplot(fila,columna,5)
    color = ('g','b','r')
    for i, c in enumerate(color):
        hist = cv2.calcHist([redimgop], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])

    plt.title("Histograma: "+operacion)
    fig.add_subplot(fila,columna,8)
    #aqui va el calculo del ecualizado
    img_to_yuv = cv2.cvtColor(redimgop,cv2.COLOR_RGB2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    equaimgOp = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2RGB)
    color = ('g','b','r')
    for i, c in enumerate(color):
        hist = cv2.calcHist([equaimgOp], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
        
    plt.title("Histograma img Ecualizada:"+operacion)

    fig.add_subplot(fila,columna,11)
    plt.imshow(equaimgOp)
    plt.axis('off')
    plt.title("Imagen Ecualizada: "+operacion)
    plt.show()

operacion="Suma"
Redimgop=cv2.add(imagen1,imagen2)
MostrarImagenes(operacion,imagen1,imagen2,Redimgop)
plt.close()
operacion="Resta"
Redimgop=cv2.subtract(imagen1,imagen2)
MostrarImagenes(operacion,imagen1,imagen2,Redimgop)
plt.close()
operacion="Multiplicacion"
Redimgop=cv2.multiply(imagen1,imagen2)
MostrarImagenes(operacion,imagen1,imagen2,Redimgop)
plt.close()
operacion="Division"
Redimgop=cv2.divide(imagen1,imagen2)
MostrarImagenes(operacion,imagen1,imagen2,Redimgop)
operacion="Raiz cuadrada"
Redimgop=imagen1
Redimgop=np.sqrt(Redimgop)
Redimgop=np.asarray(Redimgop, dtype = int)
cv2.imwrite("resultSQRT.jpg",Redimgop)
Redimgop = cv2.imread('resultSQRT.jpg', 1)
MostrarImagenes(operacion,imagen1,imagen2,Redimgop)
plt.close()
operacion="Potencia"
Redimgop=imagen1
Redimgop=np.power(Redimgop,2)
Redimgop=np.asarray(Redimgop, dtype = int)
cv2.imwrite("resultPower.jpg",Redimgop)
Redimgop = cv2.imread('resultPower.jpg', 1)
MostrarImagenes(operacion,imagen1,imagen2,Redimgop)
plt.close()
operacion="Conjuncion"
Redimgop=cv2.bitwise_and(imagen1,imagen2)
MostrarImagenes(operacion,imagen1,imagen2,Redimgop)
plt.close()
operacion="Disyuncion"
Redimgop=cv2.bitwise_or(imagen1,imagen2)
MostrarImagenes(operacion,imagen1,imagen2,Redimgop)
plt.close()
operacion="Negacion"
Redimgop=imagen1
Redimgop=image= 255-Redimgop
MostrarImagenes(operacion,imagen1,imagen2,Redimgop)
plt.close()
operacion="Translacion"
ancho = imagen1.shape[1] #columnas
alto = imagen1.shape[0] #fila
M = np.float32([[1,0,2],[0,1,2]])
Redimgop = cv2.warpAffine(img1,M,(ancho,alto))
MostrarImagenes(operacion,imagen1,imagen2,Redimgop)
plt.close()
operacion="Escalado"
Redimgop= imutils.resize(imagen1,height=400)
MostrarImagenes(operacion,imagen1,imagen2,Redimgop)
plt.close()
operacion="Rotacion"
ancho = imagen1.shape[1] #columnas
alto = imagen1.shape[0] #fila
M = cv2.getRotationMatrix2D((ancho//2,alto//2),15,1)
Redimgop = cv2.warpAffine(img1,M,(ancho,alto))
MostrarImagenes(operacion,imagen1,imagen2,Redimgop)
plt.close()
