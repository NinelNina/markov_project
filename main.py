import numpy as np

import entropy
import gauss
import markov_chains
import probability


def input_array(char_arr):
    rows = 3
    cols = 3

    array = np.empty((rows, cols))
    char_arr = ["a", "b", "c"]

    for i in range(rows):
        for j in range(cols):
            element = float(input(f"Введите элемент матрицы p({char_arr[i]}|{char_arr[j]}): "))
            array[i][j] = element

    print("Матрица вероятностей переходов:")
    print(array)

    return array


def transition_matrix_to_str(matrix, char_arr):
    n = matrix.shape[0]
    str = ""
    for i in range(n):
        for j in range(n):
            str += f"p({char_arr[i]}|{char_arr[j]}) = {matrix[i][j]}\n"
    return str


def main_func():
    char_arr = ["a", "b", "c"]

    transition_matrix = input_array(char_arr)

    solution_steps, equations_coef = markov_chains.calc_probability_of_stationary_distribution(transition_matrix, char_arr)
    steps, p = gauss.gauss_elimination(equations_coef, char_arr)
    for step in steps:
        solution_steps.append(step)

    tmp, joint_matrix = probability.calc_joint_probability(transition_matrix, p, char_arr)
    solution_steps.append(tmp)

    tmp, H_X = entropy.binary_entropy(p, char_arr)
    solution_steps.append(tmp)

    tmp, H_joint = entropy.calc_joint_entropy(joint_matrix, char_arr)
    solution_steps.append(tmp)

    tmp, H_cond = entropy.conditional_entropy(p, transition_matrix)
    solution_steps.append(tmp)

    result = f"Матрица вероятностей переходов:\n{transition_matrix_to_str(transition_matrix, char_arr)}\n"
    for step in solution_steps:
        result += step

    with open("1.txt", "w") as file:
         file.write(result)


if __name__ == '__main__':
    main_func()

