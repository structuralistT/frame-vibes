from anastruct import SystemElements


def main() -> None:
    print("Создаю Г-образную раму...")

    # Создаем расчетную схему
    ss = SystemElements()

    # Г-образная рама:
    # 1) вертикальный стержень 5 м
    # 2) горизонтальный стержень 5 м
    ss.add_element(location=[[0.0, 0.0], [0.0, 5.0]])
    ss.add_element(location=[[0.0, 5.0], [5.0, 5.0]])

    # Заделка внизу (узел 1)
    ss.add_support_fixed(node_id=1)

    # Горизонтальная сила на свободном конце (узел 3)
    ss.point_load(node_id=3, Fx=10.0)

    # Расчет
    ss.solve()

    # Показ схемы
    print("Показываю расчетную схему (закройте окно, чтобы продолжить)...")
    ss.show_structure()

    input("Нажмите Enter, чтобы построить эпюру моментов...")

    # Показ эпюры изгибающих моментов
    print("Показываю эпюру изгибающих моментов (M)...")
    ss.show_bending_moment()


if __name__ == "__main__":
    main()
