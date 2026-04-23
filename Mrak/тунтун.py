from Pynite import FEModel3D
from Pynite.Visualization import Renderer
import matplotlib.pyplot as plt  # Импортируем matplotlib для отображения эпюр

# 1. Создаем модель
frame = FEModel3D()

# 2. Параметры материала и сечения
frame.add_material('Steel', E=2.1e8, G=7.7e7, nu=0.3, rho=78.5)
frame.add_section('Rect', A=1e5, Iy=1e-4, Iz=1e-4, J=1e-4)

# 3. Добавляем узлы
frame.add_node('A', 0, 0, 0)
frame.add_node('B_bot', 5, 0, 0)
frame.add_node('D', 10, 0, 0)
frame.add_node('E_bot', 15, 0, 0)

frame.add_node('M_node', 0, 4, 0)
frame.add_node('C', 2.5, 4, 0)
frame.add_node('B_top', 5, 4, 0)
frame.add_node('F', 10, 4, 0)
frame.add_node('E_top', 15, 4, 0)
frame.add_node('P_node', 20, 4, 0)

# 4. Добавляем стержни
frame.add_member('Col_A', 'A', 'M_node', 'Steel', 'Rect')
frame.add_member('Col_B', 'B_bot', 'B_top', 'Steel', 'Rect')
frame.add_member('Col_E', 'E_bot', 'E_top', 'Steel', 'Rect')

frame.add_member('AB', 'A', 'B_bot', 'Steel', 'Rect')
frame.add_member('BD', 'B_bot', 'D', 'Steel', 'Rect')

frame.add_member('MC', 'M_node', 'C', 'Steel', 'Rect')
frame.add_member('CB', 'C', 'B_top', 'Steel', 'Rect')
frame.add_member('BF', 'B_top', 'F', 'Steel', 'Rect')
frame.add_member('FE', 'F', 'E_top', 'Steel', 'Rect')
frame.add_member('EP', 'E_top', 'P_node', 'Steel', 'Rect')

# 5. Опоры
frame.def_support('A', support_DX=True, support_DY=True, support_DZ=True, support_RX=True, support_RY=True, support_RZ=False)
frame.def_support('D', support_DX=False, support_DY=True, support_DZ=True, support_RX=True, support_RY=True, support_RZ=False)
frame.def_support('E_bot', support_DX=True, support_DY=False, support_DZ=True, support_RX=True, support_RY=True, support_RZ=False)

# 6. Нагрузки
frame.add_node_load('M_node', direction='MZ', P=-20)
frame.add_node_load('P_node', direction='FY', P=-30)
frame.add_member_dist_load('BF', direction='Fy', w1=-1, w2=-1)
frame.add_member_dist_load('FE', direction='Fy', w1=-1, w2=-1)

# 7. Расчет
frame.analyze()

# 8. Вывод реакций в опорах
print("Реакция в опоре A (FY):", frame.nodes['A'].RxnFY)
print("Реакция в опоре D (FY):", frame.nodes['D'].RxnFY)

# 9. Построение эпюр
# Мы выберем один из стержней, например, 'BF', и построим для него эпюры
member_to_plot = 'BF' 

# Для этого стержня нужно знать направление осей. 
# В PyNite для построения эпюр в локальных осях стержня используются:
# - Для моментов: 'Mz' (вокруг локальной оси Z, сильный изгиб) и 'My' (вокруг локальной оси Y, слабый изгиб)
# - Для поперечных сил: 'Fy' и 'Fz'
# - Для продольной силы: отдельный метод plot_axial()

# Поскольку ваша рама плоская и нагрузки лежат в плоскости XY (глобальной), 
# основной изгиб будет происходить вокруг локальной оси Z (Mz), 
# а поперечная сила будет вдоль локальной оси Y (Fy).

try:
    # Эпюра моментов (M) вокруг оси Z
    frame.members[member_to_plot].plot_moment('Mz', 'Combo 1', n_points=100)
    plt.title(f'Эпюра моментов Mz для стержня {member_to_plot}')
    plt.show()
    
    # Эпюра поперечных сил (Q) вдоль оси Y
    frame.members[member_to_plot].plot_shear('Fy', 'Combo 1', n_points=100)
    plt.title(f'Эпюра поперечных сил Fy для стержня {member_to_plot}')
    plt.show()
    
    # Эпюра продольных сил (N)
    frame.members[member_to_plot].plot_axial('Combo 1', n_points=100)
    plt.title(f'Эпюра продольных сил N для стержня {member_to_plot}')
    plt.show()

    print(f"Эпюры для стержня '{member_to_plot}' построены.")
    print("Обратите внимание на знаки: моменты, растягивающие нижние волокна, обычно считаются положительными.")

except Exception as e:
    print(f"Не удалось построить эпюры для стержня '{member_to_plot}'. Ошибка: {e}")
    print("Возможно, для этого стержня в выбранной комбинации нулевые усилия.")

# 10. Визуализация модели
renderer = Renderer(frame)
renderer.annotation_size = 0.5
renderer.combo_name = 'Combo 1'
renderer.render_model()