import sys

# -------------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ --------------------

def read_lp_problem(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    objective = None
    constraints = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if parts[0] == 'objective:':
            objective = list(map(float, parts[1:]))
        elif parts[0] == 'constraint:':
            if len(parts) < 3:
                raise ValueError("Invalid constraint line")
            constr_type = parts[1]
            if constr_type not in ('<=', '==', '>='):
                raise ValueError(f"Unknown constraint type: {constr_type}")
            coeffs = list(map(float, parts[2:-1]))
            rhs = float(parts[-1])
            constraints.append((constr_type, coeffs, rhs))
        else:
            raise ValueError(f"Unknown line: {line}")

    if objective is None:
        raise ValueError("Objective function not found")
    return objective, constraints


def find_pivot_column(obj_row):
    min_val = min(obj_row)
    if min_val >= -1e-10:
        return -1
    return obj_row.index(min_val)


def find_pivot_row(tableau, col_index):
    n_rows = len(tableau) - 1
    rhs = [tableau[i][-1] for i in range(n_rows)]
    col = [tableau[i][col_index] for i in range(n_rows)]
    ratios = []
    for i in range(n_rows):
        if col[i] > 1e-10:
            ratios.append(rhs[i] / col[i])
        else:
            ratios.append(float('inf'))
    if all(r == float('inf') for r in ratios):
        raise Exception("Problem is unbounded")
    return ratios.index(min(ratios))


def pivot(tableau, pivot_row, pivot_col):
    divisor = tableau[pivot_row][pivot_col]
    tableau[pivot_row] = [x / divisor for x in tableau[pivot_row]]
    n_rows = len(tableau)
    n_cols = len(tableau[0])
    for i in range(n_rows):
        if i != pivot_row:
            factor = tableau[i][pivot_col]
            for j in range(n_cols):
                tableau[i][j] -= factor * tableau[pivot_row][j]


def is_optimal(obj_row):
    return min(obj_row) >= -1e-10


# -------------------- ФАЗА I --------------------

def build_phase1_tableau(c, constraints):
    n_orig = len(c)
    var_names = [f'x{i+1}' for i in range(n_orig)]
    tableau_rows = []
    rhs_values = []
    basis = []
    artificial_indices = []

    for constr_type, coeffs, rhs in constraints:
        row = coeffs[:]
        if constr_type == '<=':
            slack_idx = len(var_names)
            var_names.append(f's{len([n for n in var_names if n.startswith("s")]) + 1}')
            row.extend([0.0] * (slack_idx - len(row)))
            row.append(1.0)
            tableau_rows.append(row)
            rhs_values.append(rhs)
            basis.append(slack_idx)

        elif constr_type == '>=':
            sur_idx = len(var_names)
            var_names.append(f'sur{len([n for n in var_names if n.startswith("sur")]) + 1}')
            art_idx = len(var_names)
            var_names.append(f'a{len([n for n in var_names if n.startswith("a")]) + 1}')
            row.extend([0.0] * (sur_idx - len(row)))
            row.append(-1.0)
            row.append(1.0)
            tableau_rows.append(row)
            rhs_values.append(rhs)
            basis.append(art_idx)
            artificial_indices.append(art_idx)

        elif constr_type == '==':
            art_idx = len(var_names)
            var_names.append(f'a{len([n for n in var_names if n.startswith("a")]) + 1}')
            row.extend([0.0] * (art_idx - len(row)))
            row.append(1.0)
            tableau_rows.append(row)
            rhs_values.append(rhs)
            basis.append(art_idx)
            artificial_indices.append(art_idx)

    n_vars = len(var_names)
    final_tableau_rows = []
    for row, rhs in zip(tableau_rows, rhs_values):
        if len(row) < n_vars:
            row.extend([0.0] * (n_vars - len(row)))
        row.append(rhs)
        final_tableau_rows.append(row)
    tableau_rows = final_tableau_rows

    w_row = [0.0] * n_vars
    for idx in artificial_indices:
        w_row[idx] = 1.0

    w_row_full = w_row + [0.0]

    for i in range(len(tableau_rows)):
        if basis[i] in artificial_indices:
            for j in range(len(w_row_full)):
                w_row_full[j] -= tableau_rows[i][j]

    tableau = tableau_rows + [w_row_full]

    is_artificial = [False] * n_orig
    for name in var_names[n_orig:]:
        is_artificial.append(name.startswith('a'))

    return tableau, var_names, basis, artificial_indices, is_artificial


def phase1_simplex(tableau, basis, artificial_indices):
    while True:
        obj_row = tableau[-1][:-1]
        if is_optimal(obj_row):
            break
        pivot_col = find_pivot_column(obj_row)
        if pivot_col == -1:
            break
        pivot_row = find_pivot_row(tableau, pivot_col)
        pivot(tableau, pivot_row, pivot_col)
        basis[pivot_row] = pivot_col

    # Проверка после завершения
    if tableau[-1][-1] < -1e-6:
        raise Exception(f"No feasible solution (W = {tableau[-1][-1]:.6g})")
    return tableau, basis


# -------------------- ПЕРЕХОД К ФАЗЕ II --------------------

def prepare_phase2_tableau(tableau1, basis1, is_artificial, c_orig, var_names1):
    n_total = len(var_names1)
    n_constraints = len(tableau1) - 1

    # Оставляем только не-искусственные столбцы
    keep_cols = [i for i in range(n_total) if not (i < len(is_artificial) and is_artificial[i])]
    new_var_names = [var_names1[i] for i in keep_cols]
    new_n_vars = len(new_var_names)

    new_tableau = []
    for row in tableau1[:-1]:
        new_row = [row[i] for i in keep_cols] + [row[-1]]
        new_tableau.append(new_row)

    # Обновляем базис
    old_to_new = {old: new for new, old in enumerate(keep_cols)}
    new_basis = []
    for b in basis1:
        if b < len(is_artificial) and is_artificial[b]:
            continue  # искусственные не входят в Фазу II
        if b in old_to_new:
            new_basis.append(old_to_new[b])

    # Восстанавливаем базис до нужной длины (редкий случай)
    while len(new_basis) < n_constraints:
        found = False
        for j in range(new_n_vars):
            if j in new_basis:
                continue
            col = [new_tableau[i][j] for i in range(n_constraints)]
            if sum(1 for x in col if abs(x - 1.0) < 1e-10) == 1 and all(abs(x) < 1e-10 or abs(x - 1.0) < 1e-10 for x in col):
                new_basis.append(j)
                found = True
                break
        if not found:
            new_basis.append(0)  # fallback

    # Целевая строка
    z_row = [0.0] * new_n_vars
    for j, name in enumerate(new_var_names):
        if name.startswith('x'):
            idx = int(name[1:]) - 1
            if idx < len(c_orig):
                z_row[j] = -c_orig[idx]

    z_row_full = z_row + [0.0]

    for i in range(n_constraints):
        var_idx = new_basis[i]
        coeff = z_row_full[var_idx]
        if abs(coeff) > 1e-10:
            for j in range(len(z_row_full)):
                z_row_full[j] -= coeff * new_tableau[i][j]

    new_tableau.append(z_row_full)
    return new_tableau, new_basis, new_var_names


def phase2_simplex(tableau, basis):
    while True:
        obj_row = tableau[-1][:-1]
        if is_optimal(obj_row):
            break
        pivot_col = find_pivot_column(obj_row)
        if pivot_col == -1:
            break
        pivot_row = find_pivot_row(tableau, pivot_col)
        pivot(tableau, pivot_row, pivot_col)
        basis[pivot_row] = pivot_col
    return tableau, basis


# -------------------- ВЫВОД РЕЗУЛЬТАТА --------------------

def extract_solution(var_names, tableau, basis):
    n_vars = len(var_names)
    n_rows = len(tableau) - 1
    solution = [0.0] * n_vars
    for i in range(n_rows):
        if i < len(basis):
            var_idx = basis[i]
            if 0 <= var_idx < n_vars:
                solution[var_idx] = tableau[i][-1]
    optimal_value = tableau[-1][-1]
    return solution, optimal_value


def print_solution(var_names, tableau, basis):
    solution, opt_val = extract_solution(var_names, tableau, basis)
    print("Optimal solution found:")
    print(f"Optimal value: {opt_val:.6g}")
    for name, val in zip(var_names, solution):
        if name.startswith('x'):
            print(f"{name} = {val:.6g}")


# -------------------- ГЛАВНАЯ ФУНКЦИЯ --------------------

def main():
    filename = "task_function.txt"
    c, constraints = read_lp_problem(filename)

    print("Reading problem...")
    tableau1, var_names1, basis1, artificial_indices, is_artificial = build_phase1_tableau(c, constraints)

    print("Solving Phase I...")
    tableau1, basis1 = phase1_simplex(tableau1, basis1, artificial_indices)

    print("Preparing Phase II...")
    tableau2, basis2, var_names2 = prepare_phase2_tableau(tableau1, basis1, is_artificial, c, var_names1)

    print("Solving Phase II...")
    tableau2, basis2 = phase2_simplex(tableau2, basis2)

    print_solution(var_names2, tableau2, basis2)


if __name__ == "__main__":
    main()