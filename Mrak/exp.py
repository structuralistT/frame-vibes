from anastruct import SystemElements

# 1. Создаем объект системы
ss = SystemElements()

# Коэффициент для перевода эпюры моментов на "растянутое волокно" 
# (в anaStruct по умолчанию может быть иначе, проверим отрисовку)
# Размеры: высота 4, пролеты 2.5, 2.5, 5, 5, 5.

# 2. Добавляем элементы (стержни)
# Вертикальные стойки
ss.add_element(location=[[0, 0], [0, 4]])      # Стойка слева (от A до угла)
ss.add_element(location=[[5, 0], [5, 4]])      # Стойка в узле B
ss.add_element(location=[[15, 0], [15, 4]])    # Стойка в узле E

# Нижний пояс
# Элемент AB (затяжка). Перед узлом B шарнир -> hinge=2 (конец элемента)
ss.add_element(location=[[0, 0], [5, 0]], hinge=3) 
ss.add_element(location=[[5, 0], [10, 0]])     # BD

# Верхний пояс (Ригель)
# Участок до шарнира C. Конец шарнирный -> hinge=2
ss.add_element(location=[[0, 4], [2.5, 4]], hinge=2)
# Участок от C до B_top. Начало шарнирное -> hinge=1
ss.add_element(location=[[2.5, 4], [5, 4]], hinge=1)
# От B_top до шарнира F. Конец шарнирный -> hinge=2
ss.add_element(location=[[5, 4], [10, 4]], hinge=2)
# От F до E_top. Начало шарнирное -> hinge=1
ss.add_element(location=[[10, 4], [15, 4]], hinge=1)
# Консоль с силой P
ss.add_element(location=[[15, 4], [20, 4]])

# 3. Устанавливаем опоры
# Опора A: шарнирно-неподвижная (id узла можно найти по координатам)
node_A = ss.find_node_id([0, 0])
ss.add_support_hinged(node_id=node_A)

# Опора D: шарнирно-подвижная (вертикальная реакция)
node_D = ss.find_node_id([10, 0])
ss.add_support_roll(node_id=node_D, direction=2) # direction 2 = фиксация по Y

# Опора E: шарнирно-подвижная (горизонтальная реакция)
node_E = ss.find_node_id([15, 0])
ss.add_support_roll(node_id=node_E, direction=1) # direction 1 = фиксация по X

# 4. Прикладываем нагрузки
# Сосредоточенный момент в точке (0, 4). По фото - по часовой стрелке.
# В anaStruct положительный момент - против часовой, значит ставим -20.
node_top_left = ss.find_node_id([0, 4])
ss.moment_load(node_id=node_top_left, Tz=-20)

# Сила P = 30 в точке (20, 4). Направлена вниз -> Fy = -30.
node_P = ss.find_node_id([20, 4])
ss.point_load(node_id=node_P, Fy=-30)

# Распределенная нагрузка q = 1 на ригеле от B до E (длина 10)
# Нагрузка действует на элементы между x=5 и x=15 на высоте y=4.
# Это элементы: [5,4]-[10,4] и [10,4]-[15,4].
# В этой версии anaStruct нет find_element_id, поэтому берём ID по порядку
# добавления элементов: [5,4]-[10,4] -> 9, [10,4]-[15,4] -> 10.
el_q1 = 8
el_q2 = 9
ss.q_load(element_id=el_q1, q=-1)
ss.q_load(element_id=el_q2, q=-1)

# 5. Расчет и вывод эпюр
ss.solve()
ss.show_structure()

ss.show_reaction_force()

# Построение графиков
print("Эпюра M (Изгибающие моменты):")
ss.show_bending_moment()

print("Эпюра Q (Поперечные силы):")
ss.show_shear_force()

print("Эпюра N (Продольные силы):")
ss.show_axial_force()