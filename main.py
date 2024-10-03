import random
import math
import time
import matplotlib.pyplot as plt
import openpyxl
import numpy as np

def remove_string_from_list(edges2, index):
    to_remove = []
    for edge in edges2:
        modificate_edges = list(map(int, edge.split()))
        if index in modificate_edges:
            to_remove.append(edge)
    for item in to_remove:
        edges2.remove(item)


def individual_is_ok(individual):
    edges2 = list(edges)
    one_indices = [i for i, val in enumerate(individual) if val == '1']
    for index in one_indices:
        remove_string_from_list(edges2, index)
    if not edges2:
        accemptability = True
    else:
        accemptability = False
    return accemptability, edges2, one_indices


def find_functions(one_indices):
    F = len(one_indices) 
    Fitness_Function = (10 * n) / (1 + F)
    return F, Fitness_Function


def decline_fitness(n):
    Fitness_Function = 0.5
    F = (10 * n) / (Fitness_Function) - 1
    return F, Fitness_Function


def change_individual(individual, edges):
    accemptability = False
    while accemptability != True:
        null_indices = np.array([i for i, val in enumerate(individual) if val == '0'])
        change_index = random.choice(null_indices)
        individual = individual[:change_index] + '1' + individual[change_index + 1:]
        accemptability, edges2, one_indices = individual_is_ok(individual)
        if accemptability == True:
            F, Fitness_Function = find_functions(one_indices)
        else:
            accemptability = False
    return F, Fitness_Function


def usefull(population, edges):
    FitnessF = []
    for individual in population:
        accemptability, edges2, one_indices = individual_is_ok(individual)
        if accemptability == True:
            F, Fitness_Function = find_functions(one_indices)
        else:
            accemptability = False
            choice = random.randint(0, 1)
            if choice == 0:
                F, Fitness_Function = decline_fitness(n)
            if choice == 1:
                F, Fitness_Function = change_individual(individual, edges)
        FitnessF.append(Fitness_Function)
    return population, FitnessF


def generate_children(population, chance):
    children = []
    parents = list(population)
    for _ in range(len(parents) // 2):
        parent1 = random.choice(parents)
        parents.remove(parent1)  # Удаление выбранного первого родителя из списка
        parent2 = random.choice(parents)
        parents.remove(parent2)  # Удаление выбранного второго родителя из списка

        i = random.randint(0, 9)  # Выбор случайной позиции i от 0 до 8

        child1 = parent1[:i + 1] + parent2[i + 1:]  # Формирование первого потомка
        child2 = parent2[:i + 1] + parent1[i + 1:]  # Формирование второго потомка


        mutation1 = random.choices((0, 1), weights=[0.4, chance])[0]
        if mutation1 == 1:
            index = random.randint(0, n - 1)
            new_numb = '1' if child1[index] == '0' else '0'
            child1 = child1[:index] + new_numb + child1[index + 1:]


        mutation2 = random.choices((0, 1), weights=[0.4, chance])[0]
        if mutation2 == 1:
            index = random.randint(0, n - 1)
            new_numb = '1' if child2[index] == '0' else '0'
            child2 = child2[:index] + new_numb + child2[index + 1:]

        children.append(child1)
        children.append(child2)

    return children


def champion(parents_and_children, p_and_ch_FitnessF):
    max_fitness = max(p_and_ch_FitnessF)
    max_index = p_and_ch_FitnessF.index(max_fitness)
    if len(new_FitnessF) != 0:
        if max_fitness >= max(new_FitnessF):
            new_population[0] = parents_and_children[max_index]
            new_FitnessF[0] = max_fitness
    else:
        new_population.append(parents_and_children[max_index])
        new_FitnessF.append(max_fitness)
    value_one=0
    for i in new_population[0]:
        if i=='1':
            value_one+=1
    print('Чемпион:', new_population[0], new_FitnessF[0],value_one)
    return new_population, new_FitnessF


def proportional_selection(parents_and_children, n, p_an_ch_Fitness_F, new_FitnessF, new_population):
    total_fitness = sum(p_an_ch_Fitness_F)
    n = min(n, len(parents_and_children))

    old_list = list(zip(parents_and_children, p_an_ch_Fitness_F))
    population_list = [(new_population[0], new_FitnessF[0])]
    print("Все особи к отбору:", parents_and_children, p_an_ch_Fitness_F, "Лучшая особь :", new_population[0])

    del old_list[parents_and_children.index(new_population[0])]

    while len(population_list) < n:
        rand_val = random.uniform(0, total_fitness)
        current_sum = 0

        for individual, individual_fitness in old_list:
            current_sum += individual_fitness
            if current_sum >= rand_val:
                population_list.append((individual, individual_fitness))
                total_fitness -= individual_fitness
                old_list.remove((individual, individual_fitness))
                break

    new_population = [individual for individual, _ in population_list]
    new_FitnessF = [individual_fitness for _, individual_fitness in population_list]

    return new_population, new_FitnessF




n = int(input('Введите количество вершин в графе'))
probability_of_edge =float(input('Введите вероятность существования ребра между вершинами : '))
p = int(n * (n - 1) / 2 * probability_of_edge)  # Вычисляем количество рёбер по вероятности
N = int(math.log(n) * 5)
k = int(input('Введите количество прогонов'))
chance = float(input('Введите вероятность мутации'))

Start = time.time()  # начало подсчета времени работы

population = []
FitnessF = []
new_population = []
new_FitnessF = []
champions_fitness = []
edges=[]
x = []


wb = openpyxl.load_workbook('edges.xlsx')
sheet = wb.active


for row in sheet.iter_rows(values_only=True, min_row=1, max_row=p, min_col=1, max_col=2):
    num1, num2 = row
    edges.append(f'{num1} {num2}')

for line in edges: 
    print(line)


population = []
FitnessF = []
new_population = []
new_FitnessF = []


for i in range(N):
    new_str = ''
    for j in range(n):
        new_str += str(random.choice([0, 1]))
    population.append(new_str)
print("Начальная популяция:", population)


tmp = 0  # счетчик номера поколения

for i in range(k + 1):
    if tmp == 0:  # вычисление допустимости и функции приспособленности для каждой особи поколения Р0
        population, FitnessF = usefull(population, edges)
    x.append(tmp)
    print('Популяция P', tmp, ':', *list(zip(population, FitnessF)))
    tmp += 1
    children = generate_children(population, chance)  # генерация детей с помощью кроссинговера с заданной вер-ю
    children, children_FitnessF = usefull(children, edges)  # проверка допустимости популяции детей и подсчет фитнес ф-и
    print('Дети', *list(zip(children, children_FitnessF)))

    parents_and_children = population + children  # len(parents_and_children)==2N, тк len(population)==N and len(children)==N
    p_and_ch_FitnessF = FitnessF + children_FitnessF  # аналогично

    # поиск чемпиона и добавление его в пустой список
    new_population, new_FitnessF = champion(parents_and_children, p_and_ch_FitnessF)
    champions_fitness.append(new_FitnessF[0])

    # удаление чемпиона из списка популяции и фитнес ф-й родителей + детей для исключения повторов
    # parents_and_children.remove(new_population[0])
    # p_and_ch_FitnessF.remove(new_FitnessF[0])

    # пропорциональный отбор
    population, FitnessF = proportional_selection(parents_and_children, n, p_and_ch_FitnessF, new_FitnessF,
                                                  new_population)
    print('Отобравшиеся особи:', *list(zip(population, FitnessF)))

plt.plot(x, champions_fitness, linestyle='-')
plt.xlabel('Поколение')
plt.ylabel('Значение фитнес-функции чемпиона')
plt.title(
    'Динамика фитнес-функции чемпионов в графе из %d вершин и %d ребер, с вероятностью мутации %f' % (n, p, chance))
# plt.title('Динамика фитнес-функции чемпионов в графе из',n,'вершин и',p,'ребер','с вероятностью мутации',chance)
end = time.time()
plt.show()

#end = time.time()
print('Время выполнения кода:', end - Start, 'секунд(ы)')
