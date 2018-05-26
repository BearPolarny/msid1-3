# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

from __future__ import division

import numpy as np


def hamming_distance(X, X_train):
    """
    :param X: zbior porownwanych obiektow N1xD
    :param X_train: zbior obiektow do ktorych porownujemy N2xD
    Funkcja wyznacza odleglosci Hamminga obiektow ze zbioru X od
    obiektow X_train. Odleglosci obiektow z jednego i drugiego
    zbioru zwrocone zostana w postaci macierzy
    :return: macierz odleglosci pomiedzy obiektami z X i X_train N1xN2
    """

    def func(row): return np.sum(np.logical_xor(X_train.toarray(), row), 1)

    return np.apply_along_axis(func, 1, X.toarray())


def sort_train_labels_knn(Dist, y):
    """
    Funkcja sortujaca etykiety klas danych treningowych y
    wzgledem prawdopodobienstw zawartych w macierzy Dist.
    Funkcja zwraca macierz o wymiarach N1xN2. W kazdym
    wierszu maja byc posortowane etykiety klas z y wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist
    :param Dist: macierz odleglosci pomiedzy obiektami z X
    i X_train N1xN2
    :param y: wektor etykiet o dlugosci N2
    :return: macierz etykiet klas posortowana wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist. Uzyc algorytmu mergesort.
    """
    
    w = Dist.argsort(kind='mergesort')

    return y[w]


def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszuch sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """

    def func(row):
        r = []
        for j in range(k):
            r.append(row[j])
        return np.bincount(r, None, 4)

    return (np.apply_along_axis(func, 1, y))/k


def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """

    nval = np.shape(y_true)[0]
    err = 0
    for i in range(nval):
        idxMax = np.argmax((p_y_x[i])[::-1])
        idx = 4-idxMax-1
        if idx != y_true[i]:
            err += 1
    return err / nval


def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: zbior danych walidacyjnych N1xD
    :param Xtrain: zbior danych treningowych N2xD
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartosci parametru k, ktore maja zostac sprawdzone
    :return: funkcja wykonuje selekcje modelu knn i zwraca krotke (best_error,best_k,errors), gdzie best_error to najnizszy
    osiagniety blad, best_k - k dla ktorego blad byl najnizszy, errors - lista wartosci bledow dla kolejnych k z k_values
    """

    H = hamming_distance(Xval, Xtrain)
    S = sort_train_labels_knn(H, ytrain)
    errors = list()
    for ki in k_values:
        errors.append(classification_error(p_y_x_knn(S, ki), yval))
    minErr = min(errors)
    minIdx = errors.index(minErr)
    return minErr, k_values[minIdx], errors


def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: etykiety dla dla danych treningowych 1xN
    :return: funkcja wyznacza rozklad a priori p(y) i zwraca p_y - wektor prawdopodobienstw a priori 1xM
    """
    return np.bincount(ytrain) / np.shape(ytrain)[0]


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: dane treningowe NxD
    :param ytrain: etykiety klas dla danych treningowych 1xN
    :param a: parametr a rozkladu Beta
    :param b: parametr b rozkladu Beta
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(x|y) zakladajac, ze x przyjmuje wartosci binarne i ze elementy
    x sa niezalezne od siebie. Funkcja zwraca macierz p_x_y o wymiarach MxD.
    """

    Theta = np.zeros((4, Xtrain.shape[1]))
    XtrainA = Xtrain.toarray()

    for k in range(4):
        for d in range(Xtrain.shape[1]):
            acc1 = np.sum((ytrain == k) * (XtrainA[:, d] == 1))
            acc2 = np.sum(ytrain == k)

            Theta[k][d] = (acc1 + a - 1) / (acc2+b+a-2)

    return Theta


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: wektor prawdopodobienstw a priori o wymiarach 1xM
    :param p_x_1_y: rozklad prawdopodobienstw p(x=1|y) - macierz MxD
    :param X: dane dla ktorych beda wyznaczone prawdopodobienstwa, macierz NxD
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla kazdej z klas z wykorzystaniem klasyfikatora Naiwnego
    Bayesa. Funkcja zwraca macierz p_y_x o wymiarach NxM.
    """

    X = X.toarray()

    def p1(line):
        mul = X * line
        mul += np.logical_not(X) - np.logical_not(X) * line
        return np.apply_along_axis(np.prod, 1, mul)

    def p2(line):
        line2 = np.multiply(line, p_y)
        return line2/np.sum(line2)

    res = np.apply_along_axis(p1, 0, np.transpose(p_x_1_y))
    return np.apply_along_axis(p2, 1, res)


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: zbior danych treningowych N2xD
    :param Xval: zbior danych walidacyjnych N1xD
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrow a do sprawdzenia
    :param b_values: lista parametrow b do sprawdzenia
    :return: funkcja wykonuje selekcje modelu Naive Bayes - wybiera najlepsze wartosci parametrow a i b. Funkcja zwraca
    krotke (error_best, best_a, best_b, errors) gdzie best_error to najnizszy
    osiagniety blad, best_a - a dla ktorego blad byl najnizszy, best_b - b dla ktorego blad byl najnizszy,
    errors - macierz wartosci bledow dla wszystkich par (a,b)
    """
    ml = estimate_a_priori_nb(ytrain)
    errors = list()
    min_err = np.inf
    min_a = None
    min_b = None
    for a in a_values:
        line = []
        for b in b_values:
            estimates = estimate_p_x_y_nb(Xtrain, ytrain, a, b)
            M = p_y_x_nb(ml, estimates, Xval)
            err = classification_error(M, yval)
            line.append(err)
            if err < min_err:
                min_err = err
                min_a = a
                min_b = b
        errors.append(line)

    return min_err, min_a, min_b, errors
