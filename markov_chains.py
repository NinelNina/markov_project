import numpy as np


def calc_probability_of_stationary_distribution(arr, char_arr):
    n = len(char_arr)
    solution_steps = []
    str_res = ''
    str_num = ''
    equations_coef = np.zeros((n + 1, n + 1))
    for k in range(n):
        str_res += 'p(X_i+1 = {}) = '.format(char_arr[k])
        for j in range(n):
            if k == j:
                equations_coef[k][j] = 1
            str_res += "p({}|{})".format(char_arr[k], char_arr[j]) + " * p(X_i = {})".format(char_arr[j])
            equations_coef[k][j] -= arr[k][j]
            str_num += str(arr[k][j]) + " * p(X_i = {})".format(char_arr[j])
            if j < n - 1:
                str_res += ' + '
                str_num += ' + '
            equations_coef[n][j] = 1
        equations_coef[k][n] = 0
        str_res += " = " + str_num + "\n"
        str_num = ""
    solution_steps.append(str_res)
    equations_coef[n][n] = 1

    str_res = "Составим систему уравнений:\n"
    solution_steps.append(str_res + print_system_of_equations(equations_coef, char_arr, n + 1, n))

    str_res = "Т. к. система избыточна, уберём одно из уравнений:\n"
    equations_coef = np.delete(equations_coef, 0, axis=0)
    solution_steps.append(str_res + print_system_of_equations(equations_coef, char_arr, n, n))
    return solution_steps, equations_coef


def print_system_of_equations(equations_coef, char_arr, n, m):
    str_res = ""
    for k in range(n):
        for j in range(m):
            if equations_coef[k][j] == 1:
                str_res += f"p({char_arr[j]})"
            else:
                str_res += f"{equations_coef[k][j]} * p({char_arr[j]})"
            if j < m - 1:
                str_res += ' + '
        if k == n - 1:
            str_res += " = 1\n"
        else:
            str_res += " = 0\n"
    return str_res