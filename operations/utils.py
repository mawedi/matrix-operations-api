import numpy as np

# Matrix addition
def add_matrix_to_matrix(first_matrix, second_matrix):
    # Matrix initialization
    rows = len(second_matrix)
    columns = len(first_matrix[0])
    result = [[ 0 for _ in range(columns)] for _ in range(rows)]

    # Calculation
    for i in range(rows):
        for j in range(columns):
            result[i][j] = first_matrix[i][j] + second_matrix[i][j]

    return result


# Matrix substraction
def substract_matrix_from_matrix(first_matrix, second_matrix):
    # Matrix initialization
    rows = len(second_matrix)
    columns = len(first_matrix[0])
    result = [[ 0 for _ in range(columns)] for _ in range(rows)]

    # Calculation
    for i in range(rows):
        for j in range(columns):
            result[i][j] = first_matrix[i][j] - second_matrix[i][j]
                            
    return result


# Matrix multiplication
def multiply_dense_dense(first_matrix, second_matrix):
    # Matrix initialization
    rows = len(first_matrix)
    columns = len(second_matrix[0])
    result = [[ 0 for _ in range(columns)] for _ in range(rows)]
    
    # Calculation
    for i in range(rows):
        for j in range(columns):
            for k in range(0, len(second_matrix)):
                result[i][j] += first_matrix[i][k] * second_matrix[k][j]
        
    return result


def multiply_upper_lower_triangular(upper_matrix, lower_matrix):
    # Matrix initialization
    rows = len(upper_matrix)
    columns = len(lower_matrix[0])
    rows_lower_matrix = len(lower_matrix)
    result = [[0 for _ in range(columns)] for _ in range(rows)]
    
    # Calculation
    for i in range(rows):
        for j in range(columns):
            for k in range(i, rows_lower_matrix):
                result[i][j] += upper_matrix[i][k] * lower_matrix[k][j]

    return result 


def multiply_upper_triangular_dense(upper_matrix, dense_matrix):
    # Matrix initialization
    rows = len(upper_matrix)
    columns = len(dense_matrix[0])
    result = [[0 for _ in range(columns)] for _ in range(rows)]

    # Calculation
    for i in range(rows):
        for j in range(columns):
            for k in range(i, len(dense_matrix)):
                result[i][j] += upper_matrix[i][k] * dense_matrix[k][j]

    return result


def multiply_lower_triangular_dense(lower_matrix, dense_matrix):
    # Matrix initialization
    rows = len(lower_matrix)
    columns = len(dense_matrix[0])
    result = [[0 for _ in range(columns)] for _ in range(rows)]

    # Calculation
    for i in range(rows):
        for j in range(columns):
            for k in range(i + 1):
                result[i][j] += lower_matrix[i][k] * dense_matrix[k][j]

    return result


def multiply_banded_lower_banded_matrix(banded_matrix, lower_banded_matrix, m_first_matrix):
    rows_banded, cols_banded = len(banded_matrix), len(banded_matrix[0])
    rows_lower_banded, cols_lower_banded = len(lower_banded_matrix), len(lower_banded_matrix[0])

    # if cols_banded != rows_lower_banded:
    #     raise ValueError("Number of columns in banded matrix must be equal to the number of rows in lower banded matrix.")

    result = [[0 for _ in range(cols_lower_banded)] for _ in range(rows_banded)]

    for i in range(rows_banded):
        for j in range(cols_lower_banded):
            for k in range(max(0, i - m_first_matrix), min(cols_banded, i + m_first_matrix + 1)):
                result[i][j] += banded_matrix[i][k] * lower_banded_matrix[k][j]

    return result


def multiply_banded_dense(band_matrix, dense_matrix):
    rows_banded, cols_band = len(band_matrix), len(band_matrix[0])
    rows_dense, cols_dense = len(dense_matrix), len(dense_matrix[0])

    result = [[0 for _ in range(cols_dense)] for _ in range(rows_banded)]

    for i in range(rows_banded):
        for j in range(cols_dense):
            for k in range(i , cols_band):
                result[i][j] += band_matrix[i][k] * dense_matrix[k][j]

    return result


def multiply_lower_banded_dense(lower_band_matrix, dense_matrix, m_first_matrix):
    # Matrix initialization
    rows_lower_band, cols_lower_band = len(lower_band_matrix), len(lower_band_matrix[0])
    rows_dense, cols_dense = len(dense_matrix), len(dense_matrix[0])
    result = [[0 for _ in range(cols_dense)] for _ in range(rows_lower_band)]

    # Calculation
    for i in range(rows_lower_band):
        for j in range(cols_dense):
            for k in range(max(0, i - m_first_matrix), i + 1):
                result[i][j] += lower_band_matrix[i][k] * dense_matrix[k][j]

    return result


def multiply_banded_dense(banded_matrix, dense_matrix, m_first_matrix):
    # Matrix initialization
    rows_banded, cols_band = len(banded_matrix), len(banded_matrix[0])
    rows_dense, cols_dense = len(dense_matrix), len(dense_matrix[0])
    result = [[0 for _ in range(cols_dense)] for _ in range(rows_banded)]
    
    # Calculation
    for i in range(0, rows_banded):
        for j in range(0, cols_band):
            for k in range(max(i - m_first_matrix, 0), i + m_first_matrix):
                result[i][j] += banded_matrix[i][k] * dense_matrix[k][j]
    
    return result


def transpose_banded_matrix(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]


def multiply_banded_matrix_transpose(banded_matrix, m):
    transpose_result = transpose_banded_matrix(banded_matrix)
    result = [[0] * len(transpose_result) for _ in range(len(banded_matrix))]

    for i in range(len(banded_matrix)):
        for j in range(len(transpose_result[0])):
            for k in range(max(0, i - m), min(len(transpose_result), i + m + 1)):
                result[i][j] += banded_matrix[i][k] * transpose_result[k][j]

    return result


def inverse_gauss_jordan(matrix):
    n = len(matrix)
    identite = np.identity(n)
    augmente = np.concatenate((matrix, identite), axis=1)

    for i in range(n):
        pivot = augmente[i][i]
        augmente[i] = augmente[i] / pivot  # Utilisation de la division entière

        for j in range(n):
            if i != j:
                coef = augmente[j][i]
                augmente[j] = augmente[j] - coef * augmente[i]

    matrix_inverse = augmente[:, n:]

    return matrix_inverse


def multiply_banded_matrix_inverse(banded_matrix, matrix_inverse, m):
    n = len(banded_matrix)
    resultat = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            for k in range(max(0, i - m), min(n, i + m + 1)):
                resultat[i][j] += banded_matrix[i][k] * matrix_inverse[k][j]

    return resultat


def multiply_lower_banded_upper_banded_matrix(lower_banded_matrix, upper_banded_matrix, r, s):
    rows_lower_band, cols_lower_band = len(lower_banded_matrix), len(lower_banded_matrix[0])
    rows_upper_band, cols_upper_band = len(upper_banded_matrix), len(upper_banded_matrix[0])

    result = [[0 for _ in range(cols_upper_band)] for _ in range(rows_lower_band)]
    
    for i in range(rows_lower_band):
        for j in range(cols_upper_band):
            for k in range(max(0, j - r, i - s), min(j + 1, i + s + 1)):
                result[i][j] += lower_banded_matrix[i][k] * upper_banded_matrix[k][j]

    return result


# Multiplication matrix per Vector
def multiply_dense_vector(dense_matrix, vector):
    # Vector initialization
    rows_dense_matrix = len(dense_matrix)
    columns_dense_matrix = len(dense_matrix[0])
    result = [[] for _ in range(rows_dense_matrix)]
    
    # Calculation
    for i in range(0, rows_dense_matrix):
        result[i][0] = [sum(dense_matrix[i][j] * vector[j]) for j in range(columns_dense_matrix)]

    return result


def multiply_upper_vector(upper_matrix, vector):
    # Vector initialization
    rows_upper_matrix = len(upper_matrix)
    columns_upper_matrix = len(upper_matrix[0])
    result = [0] * rows_upper_matrix

    # Calculation
    for i in range(rows_upper_matrix):
        result[i] = sum(upper_matrix[i][j] * vector[j] for j in range(i, columns_upper_matrix))

    return result


def multiply_lower_vector(lower_matrix, vector):
    # Vector initialization
    rows_upper_matrix = len(lower_matrix)
    result = [[] for _ in range(rows_upper_matrix)]

    # Calculation
    for i in range(rows_upper_matrix):
        result[i] = sum(lower_matrix[i][j] * vector[j] for j in range(i))

    return result


def multiply_lower_banded_vector(lower_banded, vector, m):
    # Vector initialization
    rows_lower_banded = len(lower_banded)
    length_first_case_in_matrix = rows_lower_banded - m
    length_second_case_begining_in_matrix = rows_lower_banded - m + 1
    result = [[]  for _ in range(rows_lower_banded)]

    # Calculation
    for i in range(length_first_case_in_matrix):
        result[i] = [sum(lower_banded[i][j] * vector[j]) for j in range(i)]
    
    for i in range(length_second_case_begining_in_matrix, rows_lower_banded):
        result[i] = [sum(lower_banded[i][j] * vector[j]) for j in range(i - m, i)]
    
    return result


def multiply_upper_banded_vector(upper_banded, vector, m):
    # Vector initialization
    rows_upper_banded = len(upper_banded)
    length_first_case_in_matrix = rows_upper_banded - m
    length_second_case_begining_in_matrix = rows_upper_banded - m + 1
    result = [[] for _ in range(rows_upper_banded)]

    # Calculation
    for i in range(length_first_case_in_matrix): 
        result[i] = [sum(upper_banded[i][j] * vector[j]) for j in range(i, m + i)]
    
    for i in range(length_second_case_begining_in_matrix, rows_upper_banded):
        result[i] = [sum(upper_banded[i][j] * vector[j]) for j in range(i, rows_upper_banded)]
    
    return result


# Matrix solving
def solve_upper_matrix(upper_matrix, vector):
    # Vector initialization
    rows_upper_matrix = len(upper_matrix)
    result = [[] for _ in range(rows_upper_matrix)]

    # Calculation
    for i in range(rows_upper_matrix - 1, -1, -1):
        result[i][0] = vector[i][0]

        for j in range(i + 1, rows_upper_matrix):
            result[i][0] -= upper_matrix[i][j] * vector[j]

        result[i][0] = result[i][0] / upper_matrix[i][i]

    return result


def solve_lower_matrix(lower_matrix, vector):
    # Vector initialization
    vector_rows = len(vector)
    result = [[0] for _ in range(vector_rows)]  # Initialize x as a column vector

    # Calculation
    for i in range(vector_rows):
        summation = sum(lower_matrix[i][j] * result[j][0] for j in range(i))
        result[i][0] = (vector[i] - summation) / lower_matrix[i][i]

    return result


def solve_lower_banded_matrix(matrix, vector, m):
    # Initialization
    vector_rows = len(vector)
    result = [[0] for _ in range(vector_rows)] 

    # Calculation
    for i in range(vector_rows):
        result[i][0] = vector[i]

        for j in range(max(0, i - m), i):
            result[i][0] -= matrix[i][j] * result[j][0]

        result[i][0] /= matrix[i][i]

    return result


def solve_upper_banded_matrix(upper_banded, vector, m):
    # Vector initialization
    rows_upper_banded = len(upper_banded)
    result = [[] for _ in range(rows_upper_banded)]

    # Calculation
    for i in range(rows_upper_banded - 1, -1, -1):
        result[i][0] = vector[i][0]

        for j in range(i + 1, min(i + m, rows_upper_banded)):
            result[i][0] -= upper_banded[i][j] * vector[j]
            
        result[i][0] = result[i][0] / upper_banded[i][i]

    return result


def eliminate_gauss_symmetric_dense_matrix(matrix):
    matrix_rows = len(matrix)

    for k in range(matrix_rows - 1):
        for j in range(k + 1, matrix_rows):
            matrix[k][j] /= matrix[k][k]
            for i in range(j, matrix_rows):
                matrix[i][j] -= matrix[i][k] * matrix[k][j]
    
    return matrix


def solve_symmetric_desne_matrix_gauss_elimination(matrix, vector):
    matrix_rows = len(matrix)

    # Gauss elimination
    matrix = eliminate_gauss_symmetric_dense_matrix(matrix)
    
    # Remise en forme de la matrix résultante
    for i in range(matrix_rows):
        for j in range(i + 1, matrix_rows):
            matrix[i][j] = 0.0

    # Résolution du système linéaire résultant par substitution arrière
    result = [[0.0] for _ in range(matrix_rows)]  # Initialize x as a column vector
    for i in range(add_matrix_to_matrix - 1, -1, -1):
        result[i][0] = vector[i] / matrix[i][i]
        for j in range(i + 1, matrix_rows):
            result[i][0] -= matrix[j][i] / matrix[i][i] * result[j][0]

    return result


def eliminate_gauss_symmetric_banded_matrix(banded_matrix, m):
    matrix_rows = len(banded_matrix)

    for k in range(matrix_rows - 1):
        for j in range(k + 1, min(k + m, matrix_rows)):
            banded_matrix[j][k] /= banded_matrix[k][k]
            for i in range(j, min(j + m, matrix_rows)):
                banded_matrix[i][j] -= banded_matrix[i][k] * banded_matrix[k][j]
    
    return banded_matrix


def solve_symmetric_banded_matrix_gauss_elimination(banded_matrix, vector, m):
    matrix_rows = len(banded_matrix)

    # Élimination de Gauss pour la matrice bande
    banded_matrix = eliminate_gauss_symmetric_banded_matrix(banded_matrix, m)

    # Résolution du système linéaire résultant par substitution arrière
    result = [[0.0] for _ in range(matrix_rows)]  # Initialize x as a column vector
    for i in range(matrix_rows - 1, -1, -1):
        result[i][0] = vector[i] / banded_matrix[i][i]
        for j in range(i + 1, min(i + m, matrix_rows)):
            result[i][0] -= banded_matrix[j][i] / banded_matrix[i][i] * result[j][0]

    return result


def solve_symmetric_matrix_gauss_jordan(matrix, vector):
    matrix_rows = len(matrix)

    for k in range(matrix_rows):
        matrix[k][k] = matrix[k][k] ** 0.5  # Mise à jour de la diagonale principale
        for i in range(k + 1, matrix_rows):
            matrix[i][k] /= matrix[k][k]  # Mise à jour de la colonne k
            for j in range(k + 1, matrix_rows):
                matrix[i][j] -= matrix[i][k] * matrix[j][k]  # Mise à jour de la partie supérieure

    # Remise en forme de la matrix résultante
    for i in range(matrix_rows):
        for j in range(i + 1, matrix_rows):
            matrix[i][j] = 0.0

    # Résolution du système linéaire résultant par substitution arrière
    result = [[0.0] for _ in range(matrix_rows)]  # Initialize x as a column vector
    for i in range(matrix_rows - 1, -1, -1):
        result[i][0] = vector[i] / matrix[i][i]
        for j in range(i + 1, matrix_rows):
            result[i][0] -= matrix[j][i] / matrix[i][i] * result[j][0]

    return result


def solve_dense_matrix_pivot_partiel_gauss(dense_matrix, vector):
    # Getting the rows, the columns and initialization of the vector
    rows_dense_matrix = len(dense_matrix)
    columns_dense_matrix = len(dense_matrix[0])
    vector_result = [[0] for _ in range(rows_dense_matrix)]

    # Concatenate the matrix with the vector for the operation
    used_matrix = [matrix_row + vector_row for matrix_row, vector_row in zip(dense_matrix, vector)]
    
    # Changing the rows
    for i in range(rows_dense_matrix):
        pivot_index = max(range(i, rows_dense_matrix), key=lambda k: abs(used_matrix[k][i])) # Finding the index of the max pivot
        used_matrix[i], used_matrix[pivot_index] = used_matrix[pivot_index], used_matrix[i]

        # Application of the Gauss algorithm
        pivot_value = used_matrix[i][i]
        for j in range(i + 1, rows_dense_matrix):
            factor = used_matrix[j][i] / pivot_value
            used_matrix[j][i] = 0
            for k in range(i + 1, columns_dense_matrix + 1):  # Increase the range to include the augmented column
                used_matrix[j][k] -= factor * used_matrix[i][k]
        
    # Solving the lower matrix
    for i in range(rows_dense_matrix - 1, -1, -1):
        pivot_value = used_matrix[i][i]
        vector_result[i][0] = used_matrix[i][columns_dense_matrix] / pivot_value
        for j in range(i - 1, -1, -1):
            used_matrix[j][columns_dense_matrix] -= used_matrix[j][i] * vector_result[i][0]

    return vector_result



def solve_banded_matrix_pivot_partial_gauss(banded_matrix, vector, m):
    rows_matrix = len(banded_matrix)
    vector_result = [[0] for _ in range(rows_matrix)]
    bandwidth = 2 * m + 1
        
    # Concatenate the banded matrix with the vector
    used_matrix = [row + vector_row for row, vector_row in zip(banded_matrix, vector)]

    # Changing the rows
    for i in range(rows_matrix):
        pivot_index = max(range(i, min(i + 1 + bandwidth, rows_matrix)),
                          key=lambda k: abs(used_matrix[k][i]))
        used_matrix[i], used_matrix[pivot_index] = used_matrix[pivot_index], used_matrix[i]

        # Application of the Gauss algorithm
        pivot_value = used_matrix[i][i]
        for j in range(i + 1, min(i + 1 + bandwidth, rows_matrix)):
            factor = used_matrix[j][i] / pivot_value
            used_matrix[j][i] = 0
            used_matrix[j][-1] -= factor * used_matrix[i][-1]

    # Solving the lower matrix
    for i in range(rows_matrix - 1, -1, -1):
        pivot_value = used_matrix[i][i]
        vector_result[i][0] = used_matrix[i][-1] / pivot_value
        for j in range(i - 1, max(-1, i - 1 - bandwidth), -1):
            used_matrix[j][-1] -= used_matrix[j][i] * vector_result[i][0]

    return vector_result