	************************************************************
	*							   *
	* Sistem de compresie a imaginilor bazat pe retele neurale *
	*							   *
	************************************************************

Sistemul realizeaza compresia fara pierderi a imaginilor in tonuri de gri (losseless
image compression). Implementarea presupune construirea unui model de retea neurala de tip
multi-strat cu elemente de tip perceptron (MLP) ce modeleaza comportamentul unui sistem de
codare predictiva (Predictive Coding). Codarea predictiva presupune ca un pixel curent sa fie 
determinat pe baza unui numar de pixeli anteriori.
Astfel ca pentru aflarea oricarui pixel din imagine sunt necesari doar un numar de pixeli 
initiali pe baza carora se va face succesiv predictia. Numarul de pixeli ce vor fi astfel
retinuti scade semnificativ. Totodata fiind un algoritm fara pierderi (de tip lossless) este
necesara si stocarea erorilor dintre pixelii prezisi si valorile reale, adica o imagine de erori.
Imaginea de erori va fi dimensional identica cu cea reala (cea ce se vrea comprimata) insa va 
avea entropie mult mai mica ceea ce va facilita codarea Huffman (rata bpp - bits per pixel - mai
mica). Reteaua ce va modela algoritmul de codare predictiva este specifica imaginii ce se vrea
comprimata, astfel ca trebuie salvate si ponderile modelului, ceea ce insa nu este un impediment
din punct de vedere al dimensiunii, arhitectura retelei fiind simpla, numar de straturi redus.

Descrierea implementarii:
Algoritmul este implementat exclusiv in Python si contine doua script-uri:
	- compressionPart.py: unde se realizeaza compresia propriu-zisa
	- decompressionPart.py: unde se realizeaza si partea de decompresie (pentru relizarea
	unui sistem encoder complet)
Se va descrie in continuare script-ul de compresie.
	In codul principal (partea Main()) se citeste imaginea ce se prea comprimata. Se apeleaza
functia de codarea Huffman dupa ce imaginea a fost "serializata", pixelii au fost trimisi sub
forma unui sir de tip string. Se calculeaza rata bpp din lungimea codului Huffman rezultat raportat
la numarul de pixeli initiali. Se va avea ca reper aceasta rata bpp.
	Imaginea va fi parcursa pe ferestre de dimensiune 11x11 (dimensiune sugerata in lucrarea
de referinta [1]). Pixelul central ferestrei va fi cel a carui valoarea se vrea prezisa din cei
60 de pixeli anteriori. Pentru o parcurgere completa a tuturor pixelilor este necesara formarea
unei imagini cu padding de zerouri pentru a putea forma o fereastra pentru orice pixel din imagine.
	Pentru antrenarea retelei viitoare pixelii centrali sunt salvati intr-un numpy array 
pe post de valori de iesire (trainingLabels) la vectorii de intrare formati din cei 60 de pixeli
vecini (trainingData). Valorile se vor normaliza in intervalul [0, 1]. Se construieste modelul
de retea neurala conform arhitecturii propuse (60-32-16-8-4-2-1). Fiind o problema de regresie 
(predictia unei valori continue la iesire) se va alege o functie loss de tip eroare medie absoluta
si un optimizator Adam. Dupa antrenarea retelei se construieste imaginea de erori.
	Functia entropy() calculeaza entropia sursei dupa formula standard.
	Functia ce realizeaza codare Huffman (huffman) este preluata dintr-un cod open-source.

Rezultate:
	Bits per pixel is 7.468208312988281 bpp   (codare huffman img originala)
		512*512*7.468208312988281 = 1957746 b = 1,95 Mb
	
	Bits per pixel is 5.2710418701171875 bpp   (codare huffman img de erori)
		(512-5)*(512-5)*5.2710418701171875 = 1354648 b = 1,35 Mb


entropy(np.array(errorImage))
Out[58]: 5.346432268422008

entropy(img.reshape([-1]))
Out[59]: 7.445506719708217

Imaginea originala (512*512*8 = 2097152 b = 2,1 Mb)
Imaginea comprimata (2,05 Mb)


Studenti:
	Bizu Delia-Olivia
	Firuti Nicusor-Alexandru 
	(TAID 2019)


[1] : Image Compression Using Deep Learning, H. Kubra Cilingir, M. Ozan Tezcan, Sivaramakrishnan Sankarapandian