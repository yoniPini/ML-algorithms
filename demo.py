import numpy as np
import sys
import matplotlib.pyplot as plt


class Label:
    def __init__(self, x: np.ndarray, y: int) -> None:
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return (self.x.__str__() + "," + self.y.__str__())

    def __repr__(self):
        return str(self)

    def getX(self): return self.x
    def getY(self): return self.y


def getClosestK(x_y_list: list, x, k: int):
    k_closest_x = x_y_list[:k]
    k_closest_x.sort(key=lambda label: np.linalg.norm(label.getX(), x))
    max_distance_of_k = np.linalg.norm(k_closest_x[-1].getX(), x)

    for label in x_y_list:
        d = np.linalg.norm(label.getX(), x)
        if d < max_distance_of_k:
            k_closest_x.pop()
            k_closest_x.append(label)
            k_closest_x.sort(key=lambda label: np.linalg.norm(label.getX(), x))
            max_distance_of_k = np.linalg.norm(k_closest_x[-1].getX(), x)

    return k_closest_x


def KNN(x_y_list: list, x_test: np.ndarray, k: int):
    y_test = []
    for x in x_test:
        k_closest_x = getClosestK(x_y_list, x, k)

        a = [0, 0, 0]
        for label in k_closest_x:
            a[label.getY()] += 1

        y = 0
        if a[0] > a[1] and a[0] > a[2]:
            y = 0
        if a[1] > a[0] and a[1] > a[2]:
            y = 1
        else:
            y = 2

        x_y_list.append(Label(x, y))
        y_test.append(y)
    return y_test


def main():
    train_x_path_fname, train_y_path_fname, test_x_fname, out_fname = sys.argv[
        1], sys.argv[2], sys.argv[3], sys.argv[4]
    x_train = np.loadtxt(train_x_path_fname, delimiter=",")
    y_train = np.loadtxt(train_y_path_fname, delimiter=",")
    y_train = y_train.astype(int)
    x_test = np.loadtxt(test_x_fname, delimiter=",")

    try:
        outFile = open(out_fname, "w")

        min_l = np.amin(x_train, 0)
        max_l = np.amax(x_train, 0)

        label_list = []
        for i in range(5):
            label_list.append(Label(x_train[i], y_train[i]))

        print("label_list")
        print(label_list.__str__())
        l = label_list.copy()
        np.random.shuffle(l)
        print(l.__str__())
        print(label_list.__str__())
        return
        a = KNN(label_list, nor_x_train, 3)
        b = y_train == a
        print(np.count_nonzero(b == 1) / 240)

        a = KNN(label_list, nor_x_train, 5)
        outFile.write(KNN(label_list, nor_x_test, 5).__str__())
        b = y_train == a
        print(np.count_nonzero(b == 1) / 240)

        return

    except:
        pass

    finally:
        if outFile:
            outFile.close()


main()
