import numpy as np

from .exceptions import (
    DivisionByZeroException,
    PositiveMatrixException,
    ConvergenceMatrixException,
    FundamentalMinorsIncludeZero,
    SingularMatrixException
)

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
            for k in range(max(i - m_first_matrix, 0), min(i + m_first_matrix + 1, rows_banded)):
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


def multiply_row(matrix, row_index, scalar):
    matrix[row_index] = [element * scalar for element in matrix[row_index]]


def inverse_gauss_jordan(matrix):
    matrix_rows = len(matrix)
    identite = np.identity(matrix_rows)
    augmente = np.concatenate((matrix, identite), axis=1)
    column_matrix = len(augmente[0])

    for i in range(matrix_rows):
        r = i
        for j in range(i, matrix_rows):
            if abs(augmente[i][i] < abs(augmente[j][i])):
                r = j
        
        if r != i:
            augmente[i], augmente[r] = augmente[r], augmente[i]
        
        pivot = 1 / augmente[i][i]
        multiply_row(augmente, i, pivot)
        for j in range(matrix_rows):
            if j != i:
                pivot = augmente[j][i]

                for k in range(column_matrix):
                    augmente[j][k] = augmente[j][k] - pivot * augmente[i][k]
        
        inverse_matrix = [row[(-column_matrix // 2):] for row in augmente]
        return inverse_matrix


def multiply_banded_matrix_inverse(banded_matrix, matrix_inverse, m):
    n = len(banded_matrix)
    resultat = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            for k in range(max(0, i - m), min(n, i + m + 1)):
                resultat[i][j] += banded_matrix[i][k] * matrix_inverse[k][j]

    return resultat


def multiply_lower_banded_upper_banded_matrix(lower_banded_matrix, upper_banded_matrix, s, r):
    rows_lower_band, cols_lower_band = len(lower_banded_matrix), len(lower_banded_matrix[0])
    rows_upper_band, cols_upper_band = len(upper_banded_matrix), len(upper_banded_matrix[0])

    result = [[0 for _ in range(cols_upper_band)] for _ in range(rows_lower_band)]
    
    for i in range(rows_lower_band):
        for j in range(cols_upper_band):
            for k in range(max(0, i - s, j - r), min(j + 1, i + s + 1)):
                result[i][j] += lower_banded_matrix[i][k] * upper_banded_matrix[k][j]
    
    return result


# Multiplication matrix per Vector
def multiply_dense_vector(dense_matrix, vector):
    # Vector initialization
    rows_dense_matrix = len(dense_matrix)
    columns_dense_matrix = len(dense_matrix[0])
    result = [[0.0] for _ in range(rows_dense_matrix)]
    
    # Calculation
    for i in range(0, rows_dense_matrix):
        for j in range(columns_dense_matrix):
            result[i][0] += dense_matrix[i][j] * vector[j][0]

    return result


def multiply_upper_vector(upper_matrix, vector):
    # Vector initialization
    rows_upper_matrix = len(upper_matrix)
    columns_upper_matrix = len(upper_matrix[0])
    result = [[0.0] for _ in range(rows_upper_matrix)]

    # Calculation
    for i in range(rows_upper_matrix):
        for j in range(i, columns_upper_matrix):
            result[i][0] += upper_matrix[i][j] * vector[j][0]

    return result


def multiply_lower_vector(lower_matrix, vector):
    # Vector initialization
    rows_upper_matrix = len(lower_matrix)
    result = [[0.0] for _ in range(rows_upper_matrix)]

    # Calculation
    for i in range(rows_upper_matrix):
        result[i][0] = sum(lower_matrix[i][j] * vector[j][0] for j in range(i))

    return result


def multiply_lower_banded_vector(lower_banded, vector, m):
    # Vector initialization
    rows_lower_banded = len(lower_banded)
    length_first_case_in_matrix = rows_lower_banded - m
    length_second_case_begining_in_matrix = rows_lower_banded - m + 1
    result = [[0.0]  for _ in range(rows_lower_banded)]

    # Calculation
    for i in range(length_first_case_in_matrix):
        for j in range(i):
            result[i][0] += lower_banded[i][j] * vector[j][0]
    
    for i in range(length_second_case_begining_in_matrix, rows_lower_banded):
        for j in range(i - m, i):
            result[i][0] += lower_banded[i][j] * vector[j][0]
    
    return result


def multiply_upper_banded_vector(upper_banded, vector, m):
    # Vector initialization
    rows_upper_banded = len(upper_banded)
    length_first_case_in_matrix = rows_upper_banded - m
    length_second_case_begining_in_matrix = rows_upper_banded - m + 1
    result = [[0.0] for _ in range(rows_upper_banded)]

    # Calculation
    for i in range(length_first_case_in_matrix): 
        for j in range(i, m + i):
            result[i][0] += upper_banded[i][j] * vector[j][0]
    
    for i in range(length_second_case_begining_in_matrix, rows_upper_banded):
        for j in range(i, rows_upper_banded):
            result[i] += upper_banded[i][j] * vector[j][0]

    return result


# Matrix solving
def solve_upper_matrix(upper_matrix, vector):
    # Vector initialization
    rows_upper_matrix = len(upper_matrix)
    result = [[0.0] for _ in range(rows_upper_matrix)]
    
    # Calculation
    for i in range(rows_upper_matrix - 1, -1, -1):
        result[i][0] = vector[i][0]
        
        for j in range(i + 1, rows_upper_matrix):
            result[i][0] -= upper_matrix[i][j] * result[j][0]

        try:        
            result[i][0] = result[i][0] / upper_matrix[i][i]

        except ZeroDivisionError:
            raise DivisionByZeroException({"message": "La matrice admet un zero dans la diagonale."})
        
    return result


def solve_lower_matrix(lower_matrix, vector):
    # Vector initialization
    vector_rows = len(vector)
    result = [[0.0] for _ in range(vector_rows)]  # Initialize x as a column vector

    # Calculation
    for i in range(vector_rows):
        summation = sum(lower_matrix[i][j] * result[j][0] for j in range(i))
        try:
            result[i][0] = (vector[i][0] - summation) / lower_matrix[i][i]

        except ZeroDivisionError:
            raise DivisionByZeroException({"message": "La matrice admet un zero dans la diagonale."})
        
    return result


def solve_lower_banded_matrix(matrix, vector, m):
    # Initialization
    vector_rows = len(vector)
    result = [[0] for _ in range(vector_rows)] 

    # Calculation
    for i in range(vector_rows):
        result[i][0] = vector[i][0]

        for j in range(max(0, i - m), i):
            result[i][0] -= matrix[i][j] * result[j][0]

        try:
            result[i][0] = result[i][0] / matrix[i][i]
        
        except DivisionByZeroException:
            raise DivisionByZeroException({"message": "La matrice admet un zero dans la diagonale."})
    
    return result


def solve_upper_banded_matrix(upper_banded, vector, m):
    # Vector initialization
    rows_upper_banded = len(upper_banded)
    result = [[0.0] for _ in range(rows_upper_banded)]

    # Calculation
    for i in range(rows_upper_banded - 1, -1, -1):
        result[i][0] = vector[i][0]

        for j in range(i + 1, min(i + m + 1, rows_upper_banded)):
            result[i][0] -= upper_banded[i][j] * result[j][0]
        
        try:
            result[i][0] = result[i][0] / upper_banded[i][i]

        except DivisionByZeroException:
            raise DivisionByZeroException({"message": "La matrice admet un zero dans la diagonale."})
        
    return result


def eliminate_gauss_symmetric_dense_matrix(matrix):
    matrix_rows = len(matrix)
    
    for k in range(matrix_rows - 1):
        for j in range(k + 1, matrix_rows):
            try:
                matrix[k][j] = matrix[k][j] / matrix[k][k]

            except ZeroDivisionError:
                raise DivisionByZeroException({"message": "La matrice admet un zero dans la diagonale."})
            
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
    for i in range(matrix_rows - 1, -1, -1):
        try:
            result[i][0] = vector[i][0] / matrix[i][i]
            for j in range(i + 1, matrix_rows):
                result[i][0] -= matrix[j][i] / matrix[i][i] * result[j][0]

        except ZeroDivisionError:
                raise DivisionByZeroException({"message": "La matrice admet un zero dans la diagonale."})
        
    return result


def eliminate_gauss_symmetric_banded_matrix(banded_matrix, m):
    matrix_rows = len(banded_matrix)

    for k in range(matrix_rows - 1):
        for j in range(k + 1, min(k + m, matrix_rows)):
            try:
                banded_matrix[j][k] /= banded_matrix[k][k]
            
            except ZeroDivisionError:
                raise DivisionByZeroException({"message": "La matrice admet un zero dans la diagonale."})

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
        try:
            result[i][0] = vector[i][0] / banded_matrix[i][i]
            for j in range(i + 1, min(i + m, matrix_rows)):
                result[i][0] -= banded_matrix[j][i] / banded_matrix[i][i] * result[j][0]
        
        except ZeroDivisionError:
                raise DivisionByZeroException({"message": "La matrice admet un zero dans la diagonale."})

    return result


def solve_symmetric_banded_matrix_gauss_jordan(matrix, vector, m):
    if not positive_condition(matrix):
        raise PositiveMatrixException({"message": "La matrice n'est pas définie positive."}) 

    matrix_rows = len(matrix)

    for k in range(matrix_rows):
        # Pivotisation : Diviser la ligne k par l'élément a(kk)
        pivot = matrix[k][k]
        matrix[k, k:min(k + m, matrix_rows)] = matrix[k, k:min(k + m, matrix_rows)] / pivot

        # Élimination
        for i in range(max(0, k - m), min(k + m, matrix_rows)):
            if i != k:
                factor = matrix[i, k]
                matrix[i, k:min(k + m, matrix_rows)] -= factor * matrix[k, k:min(k + m, matrix_rows)]

    # Récupérer le vecteur solution sous forme de colonne
    result = matrix[:, -1]
    return result


def positive_condition(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    
    except np.linalg.LinAlgError:
        return False


def solve_symmetric_matrix_gauss_jordan(matrix, vector):
    if not positive_condition(matrix):
        raise PositiveMatrixException({"message": "La matrice n'est pas définie positive."}) 

    matrix_rows = len(matrix)

    # Augmenter la matrice avec le vecteur b
    for i in range(matrix_rows):
        matrix[i].append(vector[i][0])

    for k in range(matrix_rows):
        # Pivotisation : Diviser la ligne k par l'élément a(kk)
        pivot = matrix[k][k]
        for j in range(k, matrix_rows + 1):
            matrix[k][j] /= pivot

        # Élimination
        for i in range(matrix_rows):
            if i != k:
                factor = matrix[i][k]
                for j in range(k, matrix_rows + 1):
                    matrix[i][j] -= factor * matrix[k][j]

    # Récupérer le vecteur solution sous forme de colonne
    solution = [[row[matrix_rows]] for row in matrix]

    return solution


def lu_decomposition_dense(matrix):
    matrix_rows = len(matrix)

    minors = []
    for i in range(matrix_rows):
        for j in range(matrix_rows):
            new_matrix = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
            minor = np.linalg.det(new_matrix)
            minors.append(minor)

    if 0 in minors:
        raise FundamentalMinorsIncludeZero({"message": "Le mineur fondamental est zero."})
    
    # Initialisation des matrices L et U
    L = [[0.0] * matrix_rows for _ in range(matrix_rows)]
    U = [[0.0] * matrix_rows for _ in range(matrix_rows)]

    for i in range(matrix_rows):
        # La diagonale de L est composée de 1
        L[i][i] = 1.0

        # Calcul de la matrice U
        for j in range(i, matrix_rows):
            sum_upper = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = matrix[i][j] - sum_upper

        # Calcul de la matrice L
        for j in range(i+1, matrix_rows):
            sum_lower = sum(L[j][k] * U[k][i] for k in range(i))
            try:
                L[j][i] = (matrix[j][i] - sum_lower) / U[i][i]

            except ZeroDivisionError:
                raise DivisionByZeroException({"message": "La matrice admet un zero dans la diagonale."})

    return L, U


def solve_symmetric_dense_matrix_LU(matrix, vector):
    L, U = lu_decomposition_dense(matrix)
    matrix_rows = len(L)

    # Étape 1: Résoudre Ly = b pour y
    y = [[0.0] for _ in range(matrix_rows)]  # Initialize y as a column vector
    for i in range(matrix_rows):
        y[i][0] = vector[i][0] - sum(L[i][k] * y[k][0] for k in range(i))

    # Étape 2: Résoudre Ux = y pour x
    result = [[0.0] for _ in range(matrix_rows)]  # Initialize x as a column vector
    for i in range(matrix_rows-1, -1, -1):
        result[i][0] = (y[i][0] - sum(U[i][k] * result[k][0] for k in range(i+1, matrix_rows))) / U[i][i]

    return result


def lu_decomposition_banded(matrix_banded, m):
    matrix_rows = len(matrix_banded)

    minors = []
    for i in range(matrix_rows):
        for j in range(matrix_rows):
            new_matrix = np.delete(np.delete(matrix_banded, i, axis=0), j, axis=1)
            minor = np.linalg.det(new_matrix)
            minors.append(minor)

    if 0 in minors:
        raise FundamentalMinorsIncludeZero({"message": "Le mineur fondamental est zero."})
    
    # Initialisation des matrices L et U
    L = [[0.0] * matrix_rows for _ in range(matrix_rows)]
    U = [[0.0] * matrix_rows for _ in range(matrix_rows)]

    for i in range(matrix_rows):
        # La diagonale de L est composée de 1
        L[i][i] = 1.0

        # Calcul de la matrice U
        for j in range(i, min(i + m + 1, matrix_rows)):
            U[i][j] = matrix_banded[i][j] - sum(L[i][k] * U[k][j] for k in range(max(0, i - m), i))

        # Calcul de la matrice L
        for j in range(i + 1, min(i + m + 1, matrix_rows)):
            L[j][i] = (matrix_banded[j][i] - sum(L[j][k] * U[k][i] for k in range(max(0, j - m), j))) / U[i][i]

    return L, U


def solve_symmetric_banded_matrix_LU(matrix, vector, m):
    L, U = lu_decomposition_banded(matrix, m)
    matrix_rows = len(L)

    # Étape 1: Résoudre Ly = b pour y
    y = [[0.0] for _ in range(matrix_rows)]  # Initialiser y comme un vecteur colonne

    for i in range(matrix_rows):
        for j in range(max(0, i - m + 1), i):
            y[i][0] -= L[i][j] * y[j][0]
        y[i][0] += vector[i]

    # Étape 2: Résoudre Ux = y pour x
    x = [[0.0] for _ in range(matrix_rows)]  # Initialiser x comme un vecteur colonne

    for i in range(matrix_rows - 1, -1, -1):
        for j in range(i + 1, min(i + m, matrix_rows)):
            x[i][0] -= U[i][j] * x[j][0]
        x[i][0] += y[i][0] / U[i][i]

    return x


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


def cholesky_decomposition_dense_matrix(matrix):
    """
    Factorisation de Cholesky : A = LLᵀ
    """
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                # Élément diagonal
                summation = sum(L[i][k] ** 2 for k in range(j))
                L[i][j] = (matrix[i][j] - summation) ** 0.5
            else:
                # Éléments non diagonaux
                try:
                    summation = sum(L[i][k] * L[j][k] for k in range(j))
                    L[i][j] = (matrix[i][j] - summation) / L[j][j]

                except ZeroDivisionError:
                    raise DivisionByZeroException({"message": "La matrice admet un zero dans la diagonale."})
                
    # Transposer la matrice L pour obtenir Lᵀ
    LT = [[L[j][i] for j in range(n)] for i in range(n)]

    return L, LT


def solve_cholesky_dense_matrix(matrix, vector):
    """
    Résoudre le système linéaire Ax = b en utilisant la factorisation de Cholesky.
    """
    # Résoudre Ly = b en utilisant la substitution avant
    L, LT = cholesky_decomposition_dense_matrix(matrix)
    matrix_rows = len(L)

    y = [[0.0] for _ in range(matrix_rows)]  # Initialize y as a column vector
    for i in range(matrix_rows):
        try:
            y[i][0] = (vector[i][0] - sum(L[i][k] * y[k][0] for k in range(i))) / L[i][i]

        except ZeroDivisionError:
            raise DivisionByZeroException({"message": "La matrice admet un zero dans la diagonale."})
        
    # Résoudre Lᵀx = y en utilisant la substitution arrière
    result = [[0.0] for _ in range(matrix_rows)]  # Initialize x as a column vector
    for i in range(matrix_rows - 1, -1, -1):
        result[i][0] = (y[i][0] - sum(LT[i][k] * result[k][0] for k in range(i + 1, matrix_rows))) / LT[i][i]

    return result


def cholesky_decomposition_banded_matrix(banded_matrix, m):
    """
    Factorisation de Cholesky pour une matrice bande : A = LLᵀ
    """
    matrix_rows = len(banded_matrix)
    L = [[0.0] * matrix_rows for _ in range(matrix_rows)]

    for i in range(matrix_rows):
        for j in range(max(0, i - m + 1), i + 1):
            if i == j:
                # Élément diagonal
                summation = sum(L[i][k] ** 2 for k in range(max(0, j - m + 1), j))
                L[i][j] = (banded_matrix[i][j] - summation) ** 0.5
            else:
                # Éléments non diagonaux
                summation = sum(L[i][k] * L[j][k] for k in range(max(0, i - m + 1), min(j, i)))
                L[i][j] = (banded_matrix[i][j] - summation) / L[j][j]

    # Calculer la transposée de L (L^T)
    LT = [[L[j][i] for j in range(matrix_rows)] for i in range(matrix_rows)]

    return L, LT


def solve_cholesky_banded_matrix(banded_matrix, vector, m):
    """
    Résoudre le système linéaire Ax = b en utilisant la factorisation de Cholesky pour une matrice bande.
    """
    L, LT = cholesky_decomposition_banded_matrix(banded_matrix, m)
    matrix_rows = len(L)

    # Résoudre Ly = b en utilisant la substitution avant
    y = [[0.0] for _ in range(matrix_rows)]  # Initialize y as a column vector
    for i in range(matrix_rows):
        try:
            y[i][0] = (vector[i][0] - sum(L[i][k] * y[k][0] for k in range(max(0, i - m + 1), i))) / L[i][i]

        except ZeroDivisionError:
            raise DivisionByZeroException({"message": "La matrice admet un zero dans la diagonale."})
        
    # Résoudre L^Tx = y en utilisant la substitution arrière
    result = [[0.0] for _ in range(matrix_rows)]  # Initialize x as a column vector
    for i in range(matrix_rows-1, -1, -1):
        result[i][0] = (y[i][0] - sum(LT[i][k] * result[k][0] for k in range(i+1, min(i+m, matrix_rows)))) / LT[i][i]

    return result


def set_matrix_gauss_seidel(matrix):
    # Getting matrix rows and matrix columns
    matrix_rows = len(matrix)
    matrix_columns = len(matrix[0])
    
    # Inialization of jacobi matrix
    matrix_used = [[0.0 for _ in range(matrix_columns)] for _ in range(matrix_rows)]
    F = [[0.0 for _ in range(matrix_columns)] for _ in range(matrix_rows)]
    for i in range(matrix_rows):
        for j in range(0, i + 1):
            matrix_used[i][j] = matrix[i][j]
        
        for j in range(i + 1, matrix_columns):
            F[i][j] = matrix[i][j]
    
    try:
        matrix_used_inverse = np.linalg.inv(matrix_used)
    
    except np.linalg.LinAlgError:
        raise SingularMatrixException({"message": "La matrice utilise pour obtenir la matrice gauss seidel n'est pas inversible."})

    seidel_matrix = multiply_dense_dense(matrix_used_inverse, F)

    return seidel_matrix
    


def solve_gauss_seidel_with_epsilon(matrix, vector, epsilon):
    # Initialization of max, result and counter
    maximum = 0
    matrix_rows = len(matrix)
    y = [[0] for _ in range(matrix_rows)]

    # Testing the convergence of the matrix
    seidel_matrix = set_matrix_gauss_seidel(matrix)
    eigenvalue, vectors = np.linalg.eig(seidel_matrix)
    if max(eigenvalue) >= 1:
        raise ConvergenceMatrixException({"message": "La matrice est divergente."})

    # Solving matrix
    while True:
        for i in range(matrix_rows):
            s = 0

            for j in range(matrix_rows):
                if j != i:
                    s += matrix[i][j] * y[j][0]
            
            s = (vector[i][0] - s) / matrix[i][i]
            if maximum < (abs_result := abs(y[i][0] - s)):
                maximum = abs_result
            
            y[i][0] = s
        
        if maximum > epsilon:
            break

    return y


def solve_gauss_seidel_with_max_iteration(matrix, vector, max_iteration):
    # Initialization of max, result and counter
    maximum = 0
    matrix_rows = len(matrix)
    y = [[0] for _ in range(matrix_rows)]
    counter = 0

    # Testing the convergence of the matrix
    seidel_matrix = set_matrix_gauss_seidel(matrix)
    eigenvalue, vectors = np.linalg.eig(seidel_matrix)
    if max(eigenvalue) >= 1:
        raise ConvergenceMatrixException({"message": "La matrice est divergente."})

    # Solving matrix
    for k in range(max_iteration):
        for i in range(matrix_rows):
            s = 0

            for j in range(matrix_rows):
                if j != i:
                    s += matrix[i][j] * y[j][0]
            
            s = (vector[i][0] - s) / matrix[i][i]
            if maximum < (abs_result := abs(y[i][0] - s)):
                maximum = abs_result
            
            y[i][0] = s

    return y


def set_matrix_jacobi(matrix):
    # Getting matrix rows and matrix columns
    matrix_rows = len(matrix)
    matrix_columns = len(matrix[0])

    # Inialization of jacobi matrix
    jacobi_matrix = [[0.0 for _ in range(matrix_columns)] for _ in range(matrix_rows)]

    for i in range(matrix_rows):
        for j in range(matrix_columns):
            if i != j:
                jacobi_matrix[i][j] = - (matrix[i][j] / matrix[i][i])

    return jacobi_matrix


def solve_jacobi_with_epsilon(matrix, vector, epsilon):
    # Initialization of result and counter
    matrix_rows = len(matrix)
    x = [[0] for _ in range(matrix_rows)]
    y = [[0] for _ in range(matrix_rows)]

    # Testing the convergence of the matrix
    jacobi_matrix = set_matrix_jacobi(matrix)
    eigenvalue, vectors = np.linalg.eig(jacobi_matrix)
    if max(eigenvalue) >= 1:
        raise ConvergenceMatrixException({"message": "La matrice est divergente."})
       
    # solving matrix
    while True:
        for i in range(matrix_rows):
            x[i][0] = y[i][0]

        for i in range(matrix_rows):
            s = vector[i][0]

            for j in range(matrix_rows):
                if i != j:
                    s -= matrix[i][j] * x[j][0]

            y[i][0] = s / matrix[i][i]

        if max(abs(x[0] - y[0]) for y, x in zip(y, x)) > epsilon:
            break
            
    return y


def solve_jacobi_with_max_iteration(matrix, vector, max_iteration):
    # Initialization of result and counter
    matrix_rows = len(matrix)
    x = [[0] for _ in range(matrix_rows)]
    y = [[0] for _ in range(matrix_rows)]
    counter = 0

    # Testing the convergence of the matrix
    jacobi_matrix = set_matrix_jacobi(matrix)
    eigenvalue, vectors = np.linalg.eig(jacobi_matrix)
    if max(eigenvalue) >= 1:
        raise ConvergenceMatrixException({"message": "La matrice est divergente."})
       
    # solving matrix
    for k in range(max_iteration):
        for i in range(matrix_rows):
            x[i][0] = y[i][0]

        for i in range(matrix_rows):
            s = vector[i][0]

            for j in range(matrix_rows):
                if i != j:
                    s -= matrix[i][j] * x[j][0]

            y[i][0] = s / matrix[i][i]
            
    return y