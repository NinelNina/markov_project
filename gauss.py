import numpy as np


def gauss_elimination(matrix, char_arr):
    n = len(matrix)
    steps = []

    def matrix_to_str(matrix):
        return '\n'.join([' '.join([f'{x:.3f}' for x in row]) for row in matrix])

    steps.append("Исходная матрица:\n" + matrix_to_str(matrix) + "\n")

    for i in range(n):
        max_row = np.argmax(np.abs(matrix[i:, i])) + i
        matrix[[i, max_row]] = matrix[[max_row, i]]

        pivot = matrix[i, i]
        matrix[i, :] /= pivot

        steps.append(f"Шаг {i + 1}:\n" + matrix_to_str(matrix) + "\n")

        for j in range(n):
            if j != i:
                factor = matrix[j, i]
                matrix[j, :] -= factor * matrix[i, :]

                steps.append(f"Шаг {i + 1} (приведение к ступенчатому виду):\n" + matrix_to_str(matrix) + "\n")

    solutions = matrix[:, -1]
    steps.append("Решение:\n")
    for i, solution in enumerate(solutions):
        steps.append(f'p({char_arr[i]}) = {solution:.3f}\n')

    return steps, solutions
