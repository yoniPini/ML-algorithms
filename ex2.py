import numpy as np
import sys


def pack(nor_x_train: np.ndarray, y_train: np.ndarray):
    np.warnings.filterwarnings(
        'ignore', category=np.VisibleDeprecationWarning)
    label_list = np.array([[nor_x_train[0], y_train[0]]], dtype=np.ndarray)
    for i in range(len(nor_x_train)):
        label_list = np.append(
            label_list, [[nor_x_train[i], y_train[i]]], 0)
    label_list = np.delete(label_list, 0, 0)
    return label_list


def unpack(x_y_list: np.ndarray):
    if len(x_y_list) == 0:
        return None, None

    x_list = np.array([x_y_list[0][0]])
    y_list = np.array([x_y_list[0][1]])
    for i in x_y_list:
        x_list = np.append(x_list, [i[0]], 0)
        y_list = np.append(y_list, [i[1]], 0)

    x_list = np.delete(x_list, 0, 0)
    y_list = np.delete(y_list, 0, 0)
    return x_list, y_list


def k_fold_cross_validation(k: int, lis: list, index: int, x_y_list: np.ndarray, params: tuple):
    np.random.shuffle(x_y_list)
    n = len(x_y_list)
    percentage = 0.0
    for i in range(k):
        begin_k = int(i * n / k)
        end_k = int((i + 1) * n / k)
        train_set_k = np.concatenate([x_y_list[: begin_k], x_y_list[end_k:]])
        valid_set_k = x_y_list[begin_k:end_k]
        valid_set_k_x, valid_set_k_y = unpack(valid_set_k)
        answer = lis[index]((train_set_k, valid_set_k_x) + params)
        b = valid_set_k_y == answer
        percentage += np.count_nonzero(b == 1) / (n / k)

    return percentage / k


def getClosestK(x_y_list: np.ndarray, x, k: int):
    k_closest_x = (x_y_list[:k]).tolist()
    k_closest_x.sort(key=lambda label: np.linalg.norm(label[0] - x))
    max_distance_of_k = np.linalg.norm(k_closest_x[-1][0] - x)

    for label in x_y_list:
        d = np.linalg.norm(label[0] - x)
        if d < max_distance_of_k:
            k_closest_x.pop()
            k_closest_x.append(label)
            k_closest_x.sort(
                key=lambda label: np.linalg.norm(label[0] - x))
            max_distance_of_k = np.linalg.norm(k_closest_x[-1][0] - x)

    return k_closest_x


def KNN(x_y_list: np.ndarray, x_test: np.ndarray, k: int):
    y_test = []
    local_x_y_list = x_y_list.copy()
    np.random.shuffle(local_x_y_list)

    for x in x_test:
        k_closest_x = getClosestK(local_x_y_list, x, k)

        a = [0, 0, 0]
        for label in k_closest_x:
            a[label[1]] += 1

        y = 0
        if a[1] > a[0] and a[1] > a[2]:
            y = 1
        elif a[2] >= a[0] and a[2] >= a[1]:
            y = 2

        y_test.append(y)
    return y_test


def KNN_tuple(t: tuple):
    return KNN(t[0], t[1], t[2])


def normalize(t: np.ndarray, min_l: np.ndarray, max_l: np.ndarray):
    nor = np.array([])
    for i in range(len(t)):
        n = (t[i] - min_l[i]) / (max_l[i] - min_l[i])
        nor = np.append(nor, n)
    return nor


def accuracy_epoch(w: np.ndarray, x_y_list: np.ndarray):
    y_test = []
    x_list, y_list = unpack(x_y_list)
    for x in x_list:
        y_test.append(np.argmax(np.dot(w, x)))

    b = (y_list == y_test)
    return np.count_nonzero(b == 1) / len(y_test)


def perceptron(x_y_list: np.ndarray, x_test: np.ndarray, epochs: int, rate: float):
    y_test = []
    local_x_y_list = x_y_list.copy()
    w = np.zeros((3, len(x_test[0])))
    percentage = accuracy_epoch(w, x_y_list)
    best_w = w.copy()

    # training
    for e in range(epochs):
        np.random.shuffle(local_x_y_list)
        for label in local_x_y_list:
            y_hat = np.argmax(np.dot(w, label[0]))
            if y_hat != label[1]:
                w[label[1], :] += rate * label[0]
                w[y_hat, :] -= rate * label[0]

        t = accuracy_epoch(w, x_y_list)
        if t > percentage:
            best_w = w.copy()
            percentage = t

    # test
    for x in x_test:
        y_test.append(np.argmax(np.dot(best_w, x)))
    return y_test


def perceptron_tuple(t: tuple):
    return perceptron(t[0], t[1], t[2], t[3])


def get_y_r(y: int, w: np.ndarray, x: np.ndarray):
    y_r = 0
    max = np.dot(w[0], x)
    for i in range(len(w)):
        if i != y and np.dot(w[i], x) > max:
            y_r = i
    return y_r


def PA(x_y_list: np.ndarray, x_test: np.ndarray, epochs: int):
    y_test = []
    local_x_y_list = x_y_list.copy()
    w = np.zeros((3, len(x_test[0])))
    percentage = accuracy_epoch(w, x_y_list)
    best_w = w.copy()

    # training
    for e in range(epochs):
        # maybe should remove this line from loop
        np.random.shuffle(local_x_y_list)
        for label in local_x_y_list:
            y_r = get_y_r(label[1], w, label[0])
            loss = 1 - np.dot(w[label[1]], label[0]) + \
                np.dot(w[y_r], label[0])

            if loss > 0:
                tao = loss / (2 * np.linalg.norm(label[0])**2)
                w[label[1], :] = w[label[1], :] + tao * label[0]
                w[y_r, :] = w[y_r, :] - tao * label[0]

        t = accuracy_epoch(w, x_y_list)
        if t > percentage:
            best_w = w.copy()
            percentage = t

    # test
    for x in x_test:
        y_test.append(np.argmax(np.dot(best_w, x)))
    return y_test


def PA_tuple(t: tuple):
    return PA(t[0], t[1], t[2])


def SVM(x_y_list: np.ndarray, x_test: np.ndarray, epochs: int, rate: float, lamb: float):
    y_test = []
    local_x_y_list = x_y_list.copy()
    w = np.zeros((3, len(x_test[0])))
    percentage = accuracy_epoch(w, x_y_list)
    best_w = w.copy()

    # training
    for e in range(epochs):
        np.random.shuffle(local_x_y_list)
        for label in local_x_y_list:
            y_r = get_y_r(label[1], w, label[0])
            loss = 1 - np.dot(w[label[1]], label[0]) + \
                np.dot(w[y_r], label[0])
            for row in range(len(w)):
                w[row] *= (1 - lamb * rate)
            if loss > 0:
                w[label[1]] += rate * label[0]
                w[y_r] -= rate * label[0]

        t = accuracy_epoch(w, x_y_list)
        if t > percentage:
            best_w = w.copy()
            percentage = t

    # test
    for x in x_test:
        y_test.append(np.argmax(np.dot(best_w, x)))
    return y_test


def SVM_tuple(t: tuple):
    return SVM(t[0], t[1], t[2], t[3], t[4])


def main():
    # getting the data from files
    train_x_path_fname, train_y_path_fname, test_x_fname, out_fname = sys.argv[
        1], sys.argv[2], sys.argv[3], sys.argv[4]
    x_train = np.loadtxt(train_x_path_fname, delimiter=",")
    y_train = np.loadtxt(train_y_path_fname, delimiter=",")
    x_test = np.loadtxt(test_x_fname, delimiter=",")
    y_train = y_train.astype(int)

    try:
        outFile = open(out_fname, "w")

        # for min_max normalization
        min_l = np.amin(x_train, 0)
        max_l = np.amax(x_train, 0)

        # normalizing train data
        nor_x_train = np.array([min_l])
        for i in x_train:
            nor_x_train = np.append(
                nor_x_train, [normalize(i, min_l, max_l)], 0)
        nor_x_train = np.delete(nor_x_train, 0, 0)

        # normalizing test data
        nor_x_test = np.array([min_l])
        for i in x_test:
            nor_x_test = np.append(
                nor_x_test, [normalize(i, min_l, max_l)], 0)
        nor_x_test = np.delete(nor_x_test, 0, 0)

        # after normalizing add to test and train a column of 1 for bias
        nor_x_test = np.hstack(
            (nor_x_test, np.ones((nor_x_test.shape[0], 1))))
        nor_x_train = np.hstack(
            (nor_x_train, np.ones((nor_x_train.shape[0], 1))))
        label_list = pack(nor_x_train, y_train)

        knn = KNN(label_list, nor_x_test, 3)
        perc = perceptron(label_list, nor_x_test, 200, 0.05)
        svm = SVM(label_list, nor_x_test, 100, 0.1, 0.01)
        pa = PA(label_list, nor_x_test, 200)

        for i in range(len(nor_x_test)):
            outFile.write(
                f"knn: {knn[i]}, perceptron: {perc[i]}, svm: {svm[i]}, pa: {pa[i]}\n")

        """
        these are vars I used for the checkings parameters below:
            lis_function = [KNN_tuple, perceptron_tuple, SVM_tuple, PA_tuple]
            index = 3

            how_many = 200
            accuracy_average = 0.0
            params_l_t = [(100,), (120,),
                        (150,), (200,)]
            best_p_l_t = [(3,), (200, 0.05), (100, 0.1, 0.01), (200,)]
        """

        # first check - checking the whole training set
        """
        for i in range(len(params_l_t)):
            accuracy_average = 0.0
            for j in range(how_many):
                result = y_train == PA(
                    label_list, nor_x_train, params_l_t[i][0])
                accuracy_average += np.count_nonzero(result) / 240
            print(
                f"first check for {params_l_t[i]}:", accuracy_average / how_many)
        """

        # second check - checking validation set of 60 samples
        """
        tempo_label_list = label_list.copy()
        for i in range(len(params_l_t)):
            accuracy_average = 0.0
            for j in range(how_many):
                np.random.shuffle(tempo_label_list)
                x_demo, y_demo = unpack(tempo_label_list[180:])
                result = y_demo == PA(
                    tempo_label_list[:180], x_demo, params_l_t[i][0])
                accuracy_average += np.count_nonzero(result) / 60
            print(
                f"second check for {params_l_t[i]}:", accuracy_average / how_many)
        """

        # third check - checking k fold cross for k=5 and k=10
        """
        for i in range(len(params_l_t)):
            accuracy_average = 0.0
            k1 = 5
            for j in range(int(how_many / k1)):
                accuracy_average += k_fold_cross_validation(
                    k1, lis_function, index, label_list, params_l_t[i])
            print(f"third check for k = 5 for {params_l_t[i]}:",
                  accuracy_average / int(how_many / k1))

            accuracy_average = 0.0
            k2 = 10
            for j in range(int(how_many / k2)):
                accuracy_average += k_fold_cross_validation(
                    k2, lis_function, index, label_list, params_l_t[i])
            print(f"third check for k = 10 for {params_l_t[i]}:",
                  accuracy_average / int(how_many / k2))
        """

        # test set - testing the best parameters on the test set
        """
        y_test = np.loadtxt("test_y.txt", delimiter=",")
        y_test = y_test.astype(int)
        best_index = 3

        accuracy_average = 0.0
        for j in range(how_many):
            result = y_test == PA(
                label_list, nor_x_test, params_l_t[best_index][0])
            accuracy_average += np.count_nonzero(result) / 60
        print(
            f"test for {params_l_t[best_index]}:", accuracy_average / how_many)
        """

        # test set for all best params for each algorithem
        """
        y_test = np.loadtxt("test_y.txt", delimiter=",")
        y_test = y_test.astype(int)
        how_many = 10

        for i in range(4):
            accuracy_average = 0.0
            for j in range(how_many):
                result = y_test == lis_function[i](
                    (label_list, nor_x_test) + best_p_l_t[i])
                accuracy_average += np.count_nonzero(result) / 60
            print(f"test for {lis_function[i]} with params {best_p_l_t[i]}:",
                  accuracy_average / how_many)
        """
    except Exception as e:
        print(e.__traceback__())  # shoulb be removes when submmitting

    finally:
        if outFile:
            outFile.close()


main()
