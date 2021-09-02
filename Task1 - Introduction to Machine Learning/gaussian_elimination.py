def multiply_scalar_by_vector(n, v):
    return [el * n for el in v]


def subtract_vectors(v2, v1):
    return [el2 - el1 for el2, el1 in zip(v2, v1)]


def gaussian_elimination(matrix):
    # must make all rows under diagonal equal to zero (always size - 1 rows)
    size = len(matrix)
    for col in range(size - 1):
        for row in range(col + 1, size):
            matrix[row] = subtract_vectors(matrix[row], multiply_scalar_by_vector((matrix[row][col] / matrix[col][col]),
                                                                                  matrix[col]))
    const = []
    for row in range(size - 1, -1, -1):
        sum = 0
        col = size - 1
        for constant in const:
            sum += constant * matrix[row][col]
            col -= 1
        const.append((matrix[row][size] - sum) / matrix[row][col])
    for index in range(len(const) - 1, -1, -1):
        print(f"x{size - index} = {const[index]}")


if __name__ == '__main__':
    gaussian_elimination([[300, -100, 0, 0, 0, 20000],
                          [-100, 200, -100, 0, 0, 0],
                          [0, -100, 200, -100, 0, 0],
                          [0, 0, -100, 200, -100, 0],
                          [0, 0, 0, -100, 300, 80000]])
