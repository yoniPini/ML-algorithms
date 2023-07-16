import numpy as np
import pathlib
import os
import re

color_W = '\033[0m'  # white (normal)
color_R = '\033[31m'  # red
color_G = '\033[32m'  # green
color_O = '\033[33m'  # orange
color_B = '\033[34m'  # blue
color_P = '\033[35m'  # purple

dir = os.path.dirname(os.path.abspath(__file__))


def create_files():
    try:
        from sklearn.utils import shuffle
    except ImportError:
        print('Install sklearn before: "pip install sklearn"')

    train_x_data = np.loadtxt(os.path.join(dir, "original", 'train_x.txt'), delimiter=",", dtype=str)
    train_y_data = np.loadtxt(os.path.join(dir, "original", 'train_y.txt'), delimiter=",", dtype=int)

    test_x_data = np.loadtxt(os.path.join(dir, "original", 'test_x.txt'), delimiter=",", dtype=str)
    test_y_data = np.loadtxt(os.path.join(dir, "original", "test_y.txt"), delimiter=",", dtype=int)

    test_length = train_x_data.shape[0]
    all_x = np.concatenate((train_x_data, test_x_data))
    all_y = np.concatenate((train_y_data, test_y_data))

    example_count = all_x.shape[0]

    for i in range(number_of_files):
        shuffled_x, shuffled_y = shuffle(all_x, all_y)

        new_train_x_data = shuffled_x[:test_length]
        new_train_y_data = shuffled_y[:test_length]

        new_test_x_data = shuffled_x[test_length:]
        new_test_y_data = shuffled_y[test_length:]

        np.savetxt(os.path.join(dir, "created", "train_x_{}.txt".format(i)), new_train_x_data, delimiter=",", fmt="%s")
        np.savetxt(os.path.join(dir, "created", "train_y_{}.txt".format(i)), new_train_y_data, fmt="%s")
        np.savetxt(os.path.join(dir, "created", "test_x_{}.txt".format(i)), new_test_x_data, delimiter=",", fmt="%s")
        np.savetxt(os.path.join(dir, "created", "test_y_{}.txt".format(i)), new_test_y_data, fmt="%s")
        with open(os.path.join(dir, "created", "correct_out_{}.txt".format(i)), "w") as f:
            for y in new_test_y_data:
                f.write("knn: {0}, perceptron: {0}, svm: {0}, pa: {0}\n".format(y))


def compare_files(my_out, correct_out, i):
    with open(my_out, "r") as f:
        my_out_lines = f.readlines()
    with open(correct_out, "r") as f:
        correct_out_lines = f.readlines()

    total_size = len(my_out_lines)

    knn_error = perceptron_error = svm_error = pa_error = 0
    for l1, l2 in zip(my_out_lines, correct_out_lines):
        import parse
        my_knn, my_perceptron, my_svm, my_pa = parse.parse("knn: {0}, perceptron: {1}, svm: {2}, pa: {3}\n", l1)
        correct_knn, correct_perceptron, correct_svm, correct_pa = parse.parse(
            "knn: {0}, perceptron: {1}, svm: {2}, pa: {3}\n", l2)

        if correct_knn != my_knn:
            knn_error += 1
        if correct_perceptron != my_perceptron:
            perceptron_error += 1
        if correct_svm != my_svm:
            svm_error += 1
        if correct_pa != my_pa:
            pa_error += 1

    knn_accuracy = (total_size - knn_error) / total_size * 100
    perceptron_accuracy = (total_size - perceptron_error) / total_size * 100
    svm_accuracy = (total_size - svm_error) / total_size * 100
    pa_accuracy = (total_size - pa_error) / total_size * 100
    avg_accuracy = (knn_accuracy + perceptron_accuracy + svm_accuracy + pa_accuracy) / 4
    color = color_G if avg_accuracy >= 93 else color_R
    print(f"itr {i} you achived knn: {knn_accuracy} % perceptron: {perceptron_accuracy} % svm: {svm_accuracy} % pa: {pa_accuracy}" +
          color + f" | avg accuracy: {avg_accuracy}" + color_W)

    return np.array([knn_accuracy, perceptron_accuracy, svm_accuracy, pa_accuracy])


def test(test_fake_files=True):
    import subprocess
    results = np.zeros((number_of_files, 4))
    for i in range(number_of_files):
        my_out = os.path.abspath(os.path.join(dir, "out", "out_{}.txt".format(i)))

        if test_fake_files:
            # !!!! for testing the new creaetd files:

            train_x = os.path.abspath(os.path.join(dir, "created", "train_x_{}.txt".format(i)))
            train_y = os.path.abspath(os.path.join(dir, "created", "train_y_{}.txt".format(i)))
            test_x = os.path.abspath(os.path.join(dir, "created", "test_x_{}.txt".format(i)))
            correct_out = os.path.join(dir, "created",  "correct_out_{}.txt".format(i))
        else:
            # !!!! for testing the original files multiple times

            train_x = os.path.abspath(os.path.join(dir, "original", "train_x.txt"))
            train_y = os.path.abspath(os.path.join(dir, "original", "train_y.txt"))
            test_x = os.path.abspath(os.path.join(dir, "original", "test_x.txt"))
            correct_out = os.path.join(dir, "original",  "correct_out.txt")

        python_file = os.path.abspath(os.path.join(dir, "..", "ex2.py"))
        subprocess.call(["python3", python_file, train_x, train_y, test_x, my_out], cwd=dir)

        results[i] = compare_files(my_out, correct_out, i)

    print("avarage results:")
    knn_accuracy, perceptron_accuracy, svm_accuracy, pa_accuracy = np.average(results, axis=0)
    avg_accuracy = np.sum(np.average(results, axis=0)) / 4
    color = color_G if avg_accuracy >= 93 else color_R
    print(f"you achived knn: {knn_accuracy} % perceptron: {perceptron_accuracy} % svm: {svm_accuracy} % pa: {pa_accuracy}" +
          color + f" | avg accuracy: {avg_accuracy}" + color_W)


# Number of files to create OR number of tests on the original files
number_of_files = 50

# If you want to create fake train and test files - make sure the next lines are enables
# create_files()
# test(test_fake_files=True)

# If you want to test the original train test files (like submit) - make sure the next lines are enables
# the magic:
test(test_fake_files=True)
