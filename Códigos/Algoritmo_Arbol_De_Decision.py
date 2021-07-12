from sklearn import tree

# Creando la instancia del arbol de decision
clf = tree.DecisionTreeClassifier()

# [altura, peso, talla de zapato]
X = [[181, 80, 44],[177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
    [190, 90, 47], [175, 64, 39],
    [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['hombre', 'hombre', 'mujer', 'mujer', 'hombre', 'hombre', 'mujer', 'mujer',
     'mujer', 'hombre', 'hombre']

# Pasando datos X y Y al arbol de decision
clf = clf.fit(X, Y)

print("********* Resultados de la Clasificacion del Algoritmo de Decision *********")
print("----------------------------------------")

# Se definen los datos
dato1 = [190, 70, 43] # Clasificacion: Hombre
prediction = clf.predict([dato1])
print("Dato #1 " + str(dato1) + " Resultado: " + str(prediction))

print("----------------------------------------")
dato2 = [185, 62, 37] # Clasificacion: Mujer
prediction = clf.predict([dato2])
print("Dato #2 " + str(dato2) + " Resultado: " + str(prediction))

print("----------------------------------------")
dato3 = [160, 60, 38] # Clasificacion: Mujer
prediction = clf.predict([dato3])
print("Dato #3 " + str(dato3) + " Resultado: " + str(prediction))