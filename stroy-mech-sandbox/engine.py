"""anaStruct model build, reactions, Mohr / Vereshchagin / method of forces.

Pure numerical / FE layer: no Streamlit. Pass geometry and loads as arguments.
"""
from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from typing import Any, cast
from uuid import UUID

import numpy as np
from anastruct import SystemElements

from models import (
    ForceMethodReport,
    HingeEntry,
    KinematicAnalysisResult,
    LoadEntry,
    MemberElement,
    MohrIntegralRow,
    Node,
    NodeDisplacement,
    ReactionTableRow,
    SupportCounts,
    SupportDisplacementsMap,
    SupportDisplacementEntry,
    SupportEntry,
    UnitDirection,
    VereshchaginReport,
    coerce_element_uuid,
)


def _nodes_uuid_map(nodes: Iterable[Node]) -> dict[UUID, Node]:
    return {n.id: n for n in nodes}


def _nodes_name_map(nodes: Iterable[Node]) -> dict[str, Node]:
    return {n.name: n for n in nodes}


def _element_endpoint_names(el: MemberElement, nodes: list[Node]) -> tuple[str, str]:
    by = _nodes_uuid_map(nodes)
    n1 = by.get(el.start_node_id)
    n2 = by.get(el.end_node_id)
    if not n1 or not n2:
        return ("", "")
    return (n1.name, n2.name)


def try_solve(ss: SystemElements) -> tuple[bool, str]:
    """
    Run anaStruct linear solve. On failure (mechanism, singular stiffness, etc.)
    returns (False, human-readable message) instead of raising.
    """
    try:
        ss.solve()
        return True, ""
    except Exception as exc:
        return (
            False,
            "Решатель anaStruct не смог завершить расчёт (возможна геометрически изменяемая схема, "
            f"некорректные опоры или численная ошибка): {exc}",
        )


def build_system(
    nodes: list[Node],
    elements: list[MemberElement],
    supports: list[SupportEntry],
    loads: list[LoadEntry],
    hinges: list[HingeEntry],
    *,
    global_ei: float = 5000.0,
    global_ea: float = 1.0e9,
) -> tuple[SystemElements, list[str], dict[UUID, int]]:
    """
    Build an anastruct model from explicit lists (no session state).

    Returns:
      - model (unsolved)
      - warnings list
      - map session member UUID -> anastruct element_id
    """
    ss = SystemElements()
    warnings: list[str] = []

    if not elements:
        return ss, warnings, {}

    nodes_by_uuid = _nodes_uuid_map(nodes)
    session_to_anastruct_element: dict[UUID, int] = {}
    hinges_by_element: dict[UUID, list[str]] = {}
    for hinge in hinges:
        eid = coerce_element_uuid(hinge.get("element_id"))
        pos = hinge.get("position")
        if eid is not None and pos in ("start", "end"):
            hinges_by_element.setdefault(eid, []).append(pos)

    for element in elements:
        n1 = nodes_by_uuid.get(element.start_node_id)
        n2 = nodes_by_uuid.get(element.end_node_id)
        if not n1 or not n2:
            warnings.append(f"{str(element.id)[:8]}: пропущен, узлы не найдены.")
            continue
        if (n1.x, n1.y) == (n2.x, n2.y):
            warnings.append(f"{str(element.id)[:8]}: пропущен, одинаковые координаты узлов.")
            continue

        anastruct_element_id = len(ss.element_map) + 1
        session_to_anastruct_element[element.id] = anastruct_element_id
        spring_dict: dict[int, float] = {}
        if bool(getattr(element, "is_tie", False)):
            # Tie-rod is always pinned at both ends, regardless of explicit hinge records.
            spring_dict[1] = 0.0
            spring_dict[2] = 0.0
        else:
            for pos in hinges_by_element.get(element.id, []):
                if pos == "start":
                    spring_dict[1] = 0.0
                elif pos == "end":
                    spring_dict[2] = 0.0
        ei_val = float(global_ei)
        ea_val = float(global_ea)
        if spring_dict:
            ss.add_element(
                location=[[n1.x, n1.y], [n2.x, n2.y]],
                EA=ea_val,
                EI=ei_val,
                spring=spring_dict,
            )
        else:
            ss.add_element(
                location=[[n1.x, n1.y], [n2.x, n2.y]],
                EA=ea_val,
                EI=ei_val,
            )

    coord_to_node_id = {
        (round(node.vertex.x, 9), round(node.vertex.y, 9)): node_id for node_id, node in ss.node_map.items()
    }
    name_to_node_id: dict[str, int] = {}
    for node in nodes:
        key = (round(float(node.x), 9), round(float(node.y), 9))
        if key in coord_to_node_id:
            name_to_node_id[node.name] = coord_to_node_id[key]

    for support in supports:
        node_id = name_to_node_id.get(support["node"])
        if not node_id:
            warnings.append(f"S{support['id']}: узел '{support['node']}' не входит в текущие стержни.")
            continue

        if support["type"] == "fixed":
            ss.add_support_fixed(node_id=node_id)
        elif support["type"] == "hinged":
            ss.add_support_hinged(node_id=node_id)
        elif support["type"] == "roller":
            # anaStruct: ``add_support_roll(..., angle=...)`` — угол наклона связи, градусы.
            ss.add_support_roll(node_id=node_id, angle=float(support.get("angle", 0.0)))

    for load in loads:
        if load["type"] == "point":
            node_id = name_to_node_id.get(load["node"])
            if node_id:
                ss.point_load(node_id=node_id, Fx=load["Fx"], Fy=load["Fy"])
            else:
                warnings.append(f"L{load['id']}: сосредоточенная сила пропущена (узел вне схемы).")

        elif load["type"] == "moment":
            node_id = name_to_node_id.get(load["node"])
            if node_id:
                ss.moment_load(node_id=node_id, Tz=load["M"])
            else:
                warnings.append(f"L{load['id']}: момент пропущен (узел вне схемы).")

        elif load["type"] == "distributed":
            leu = coerce_element_uuid(load.get("element_id"))
            anastruct_element_id = session_to_anastruct_element.get(leu) if leu is not None else None
            if anastruct_element_id:
                ss.q_load(q=load["q"], element_id=anastruct_element_id)
            else:
                warnings.append(f"L{load['id']}: распределенная нагрузка пропущена (стержень вне схемы).")

    return ss, warnings, session_to_anastruct_element


def format_load_reactions_latex(reaction_rows: list[ReactionTableRow]) -> list[str]:
    """LaTeX-строки реакций в грузовом состоянии (без черты над R)."""
    lines: list[str] = []
    for row in reaction_rows:
        name = str(row["Узел"])
        fx = float(row["Fx"])
        fy = float(row["Fy"])
        mz = float(row["Mz (Tz)"])
        line = (
            rf"\text{{Узел {name}:}}\quad "
            rf"R_{{{name}x}} = {fx:.3f}\ \text{{кН}},\ R_{{{name}y}} = {fy:.3f}\ \text{{кН}}"
        )
        if row.get("Тип опоры (код)") == "fixed" or abs(mz) > 1e-8:
            line += rf",\ M_{name} = {mz:.3f}\ \text{{кН·м}}"
        lines.append(line)
    return lines


def collect_support_reaction_rows(
    ss: SystemElements, nodes: list[Node], supports: list[SupportEntry]
) -> tuple[list[ReactionTableRow], list[str]]:
    """Collect support reactions as rows for tables/cards."""
    if not supports:
        return [], []

    coord_map = _coord_to_model_node_id(ss)
    node_by_name = {n.name: n for n in nodes}

    reaction_rows: list[ReactionTableRow] = []
    missed: list[str] = []
    for support in supports:
        name = support["node"]
        node_data = node_by_name.get(name)
        if not node_data:
            missed.append(f"Узел '{name}' не найден в исходных данных.")
            continue

        key = (round(float(node_data.x), 9), round(float(node_data.y), 9))
        model_node_id = coord_map.get(key)
        if not model_node_id:
            missed.append(f"Узел '{name}' не попал в расчетную схему.")
            continue

        symbol = "".join(ch for ch in name.lower() if ch.isalnum()) or "n"
        fx, fy, mz = _reaction_components_for_node_id(ss, model_node_id)
        reaction_rows.append(
            {
                "Узел": name,
                "Индекс": symbol,
                "Тип опоры": _support_type_label(support["type"]),
                "Тип опоры (код)": support["type"],
                "Fx": fx,
                "Fy": fy,
                "Mz (Tz)": mz,
                "Направление Fx": "вправо" if fx >= 0 else "влево",
                "Направление Fy": "вверх" if fy >= 0 else "вниз",
                "Направление Mz": "против часовой" if mz >= 0 else "по часовой",
            }
        )
    return reaction_rows, missed


def compute_unit_reactions_for_settlement(
    ss_unit: SystemElements,
    nodes: list[Node],
    supports: list[SupportEntry],
    support_displacements: SupportDisplacementsMap,
) -> tuple[list[str], list[str]]:
    """
    Реакции опор в **единичном** (виртуальном) состоянии по ``ss_unit`` (после solve).

    ``support_displacements`` в сигнатуре для единообразия API с ``compute_settlement_component``;
    на состав строк реакций не влияет (выводятся все опоры из ``supports``).

    Returns:
        (latex_lines, missed_messages) — LaTeX по одной строке на опору; ``missed_messages`` — как у
        ``collect_support_reaction_rows`` (узлы вне схемы и т.п.).
    """
    _ = support_displacements
    rows, missed = collect_support_reaction_rows(ss_unit, nodes, supports)
    latex_lines: list[str] = []
    for row in rows:
        name = str(row["Узел"])
        fx = float(row["Fx"])
        fy = float(row["Fy"])
        mz = float(row["Mz (Tz)"])
        dfx = str(row.get("Направление Fx", ""))
        dfy = str(row.get("Направление Fy", ""))
        dmz = str(row.get("Направление Mz", ""))
        parts = [
            rf"\overline{{R}}_{{{name}x}} = {fx:.2f}\,\text{{кН}}\ \text{{({dfx})}}",
            rf"\overline{{R}}_{{{name}y}} = {fy:.2f}\,\text{{кН}}\ \text{{({dfy})}}",
        ]
        if str(row.get("Тип опоры (код)", "")) == "fixed" or abs(mz) > 1e-8:
            parts.append(rf"\overline{{M}}_{{{name}}} = {mz:.2f}\,\text{{кН·м}}\ \text{{({dmz})}}")
        latex_lines.append(r",\quad ".join(parts))
    return latex_lines, missed


def _coord_to_model_node_id(ss: SystemElements) -> dict[tuple[float, float], int]:
    return {(round(node.vertex.x, 9), round(node.vertex.y, 9)): node_id for node_id, node in ss.node_map.items()}


def node_name_to_model_node_id(ss: SystemElements, nodes: list[Node]) -> dict[str, int]:
    """Map editor node name -> anaStruct node id for a built (or solved) model."""
    coord_map = _coord_to_model_node_id(ss)
    name_to_id: dict[str, int] = {}
    for node in nodes:
        key = (round(float(node.x), 9), round(float(node.y), 9))
        node_id = coord_map.get(key)
        if node_id is not None:
            name_to_id[node.name] = node_id
    return name_to_id


def _ensure_reaction_forces(ss: SystemElements) -> dict[int, Any]:
    """
    Численные реакции опор после solve: в anaStruct они в ``SystemElements.reaction_forces``
    (постпроцессор ``reaction_forces()``). Имя ``get_reac_force`` встречается в старых примерах —
    при наличии метода вызываем его для заполнения полей модели.
    """
    getter = getattr(ss, "get_reac_force", None)
    if callable(getter):
        try:
            getter()
        except Exception:
            pass
    rf = getattr(ss, "reaction_forces", None) or {}
    if not rf and hasattr(ss, "post_processor"):
        try:
            ss.post_processor.reaction_forces()
        except Exception:
            pass
        rf = getattr(ss, "reaction_forces", None) or {}
    return rf


def _reaction_components_for_node_id(ss: SystemElements, node_id: int) -> tuple[float, float, float]:
    """Fx, Fy, Tz реакции опоры в глобальных осях (как в ``reaction_forces`` / get_reac_force)."""
    rf = _ensure_reaction_forces(ss)
    node_obj = rf.get(node_id)
    if node_obj is None:
        n = ss.node_map.get(node_id)
        if n is None:
            return (0.0, 0.0, 0.0)
        return (float(n.Fx), float(n.Fy), float(n.Tz))
    return (float(node_obj.Fx), float(node_obj.Fy), float(node_obj.Tz))


def _find_anastruct_element_id_for_member(
    ss: SystemElements,
    n1: Node,
    n2: Node,
) -> int | None:
    """Сопоставление стержня редактора с элементом anaStruct по координатам концов."""
    a = (round(float(n1.x), 9), round(float(n1.y), 9))
    b = (round(float(n2.x), 9), round(float(n2.y), 9))
    for eid, el in ss.element_map.items():
        p1 = (round(float(el.vertex_1.x), 9), round(float(el.vertex_1.y), 9))
        p2 = (round(float(el.vertex_2.x), 9), round(float(el.vertex_2.y), 9))
        if (p1, p2) == (a, b) or (p1, p2) == (b, a):
            return int(eid)
    return None


def _q_moment_component_scalar(x: float, L: float, qi: float, q: float) -> float:
    """Слагаемое M(x) от распределённой нагрузки (как в anaStruct ``determine_bending_moment``)."""
    if L <= 0.0:
        return 0.0
    return float(
        -((qi - q) / (6.0 * L)) * x**3
        + (qi / 2.0) * x**2
        - (((2.0 * qi) + q) / 6.0) * L * x
    )


def _q_shear_from_moment_derivative(x: float, L: float, qi: float, q: float) -> float:
    """dM_q/dx — поперечная сила от распределённой части."""
    if L <= 0.0:
        return 0.0
    return float(-((qi - q) / (2.0 * L)) * x**2 + qi * x - ((2.0 * qi) + q) / 6.0 * L)


def generate_step_by_step_analysis(
    nodes: list[Node],
    elements: list[MemberElement],
    loads: list[LoadEntry],
    supports: list[SupportEntry],
    solved_ss: SystemElements,
    *,
    hinges: list[HingeEntry] | None = None,
) -> str | list[dict[str, Any]]:
    """
    Пошаговый разбор поперечной силы и изгибающего момента (метод сечений в смысле
    аналитических Q(x), M(x) на стержне в локальной СК anaStruct).

    Реакции опор — численно из ``solved_ss.reaction_forces`` (или ``get_reac_force()`` при наличии).
    Концевые моменты T1, T2 на стержне берутся из решённой модели (учёт всей схемы и нагрузок).

    Returns:
        Строка с сообщением об ограничении / ошибке, либо список блоков
        ``{"markdown": [...], "latex": [...]}`` для последовательного вывода в UI.
    """
    hinges = hinges or []
    kin = calculate_kinematic_analysis(nodes, elements, supports, hinges)
    w = int(kin["W"])
    if w < 0:
        return "Пошаговый расчет доступен только для статически определимых систем"
    if w > 0:
        return "Пошаговый расчёт недоступен: по формуле Чебышёва W > 0 (геометрически изменяемая система)."

    if not elements:
        return "Нет стержней для пошагового разбора."

    nodes_by_uuid = _nodes_uuid_map(nodes)
    name_to_id = node_name_to_model_node_id(solved_ss, nodes)

    blocks: list[dict[str, Any]] = []
    intro_md: list[str] = [
        "**Реакции опор** (численно из расчёта; глобально: $x$ вправо, $y$ вверх):",
    ]
    for sup in supports:
        nm = str(sup.get("node", ""))
        nid = name_to_id.get(nm)
        if nid is None:
            continue
        fx, fy, tz = _reaction_components_for_node_id(solved_ss, int(nid))
        stype = _support_type_label(str(sup.get("type", "")))
        intro_md.append(
            f"- Узел **{nm}** ({stype}): "
            f"**R_x** = {fx:.3f} кН, **R_y** = {fy:.3f} кН, **M_z** = {tz:.3f} кН·м "
            f"(глобальные оси: x — вправо, y — вверх)."
        )
    blocks.append({"markdown": intro_md, "latex": []})

    for element in elements:
        if bool(getattr(element, "is_tie", False)):
            n1 = nodes_by_uuid.get(element.start_node_id)
            n2 = nodes_by_uuid.get(element.end_node_id)
            if not n1 or not n2:
                continue
            eid = _find_anastruct_element_id_for_member(solved_ss, n1, n2)
            if eid is None:
                continue
            el = solved_ss.element_map[eid]
            af = getattr(el, "axial_force", None)
            n0 = float(af[0]) if af is not None and len(af) > 0 else 0.0
            md = [
                f"**Затяжка {n1.name}–{n2.name}.** Изгиб не рассматривается; "
                f"продольная сила **N ≈ {n0:.3f} кН** (постоянно по длине, anaStruct).",
            ]
            blocks.append({"markdown": md, "latex": []})
            continue

        n1 = nodes_by_uuid.get(element.start_node_id)
        n2 = nodes_by_uuid.get(element.end_node_id)
        if not n1 or not n2:
            continue
        if (n1.x, n1.y) == (n2.x, n2.y):
            continue

        eid = _find_anastruct_element_id_for_member(solved_ss, n1, n2)
        if eid is None:
            continue
        el = solved_ss.element_map[eid]
        if getattr(el, "bending_moment", None) is None or getattr(el, "shear_force", None) is None:
            return "Выполните расчёт модели: для стержней не найдены эпюры M/Q."

        L = float(el.l)
        T1 = float(el.node_1.Tz)
        T2 = float(el.node_2.Tz)
        dT = -(T2 + T1)
        qi = float(el.all_qp_load[0]) if el.all_qp_load else 0.0
        qe = float(el.all_qp_load[1]) if el.all_qp_load else 0.0

        m0 = float(T1 + _q_moment_component_scalar(0.0, L, qi, qe))
        mL = float(T1 + dT + _q_moment_component_scalar(L, L, qi, qe))
        q0 = float(dT / L + _q_shear_from_moment_derivative(0.0, L, qi, qe)) if L > 0 else 0.0
        qL = float(dT / L + _q_shear_from_moment_derivative(L, L, qi, qe)) if L > 0 else 0.0

        dtr = dT / L if L > 1e-12 else 0.0
        ll = L if L > 1e-12 else 0.0

        md = [
            f"Участок **{n1.name}–{n2.name}** (0 ≤ x ≤ {L:.2f} м):",
        ]

        if abs(qi - qe) < 1e-12:
            if abs(qi) < 1e-12:
                q_expr = f"{dtr:.3f}"
            else:
                q_expr = f"{dtr:.3f} - {qi:.3f}·x"
        else:
            if ll > 1e-12:
                q_expr = f"{dtr:.3f} + {qi:.3f}·x - {(qi - qe) / (2 * ll):.3f}·x²"
            else:
                q_expr = f"{dtr:.3f}"

        md.append(f"$Q(x) = {q_expr}$")
        md.append(f"$Q(0) = {q0:.3f}$ кН, $Q({L:.2f}) = {qL:.3f}$ кН")

        if abs(qi - qe) < 1e-12:
            if abs(qi) < 1e-12:
                m_expr = f"{T1:.3f} + {dtr:.3f}·x"
            else:
                m_expr = f"{T1:.3f} + {dtr:.3f}·x - {qi / 2:.3f}·x²"
        else:
            if ll > 1e-12:
                m_expr = f"{T1:.3f} + {dtr:.3f}·x + {qi / 2:.3f}·x² - {(qi - qe) / (6 * ll):.3f}·x³"
            else:
                m_expr = f"{T1:.3f}"

        md.append(f"$M(x) = {m_expr}$")
        md.append(f"$M(0) = {m0:.3f}$ кН·м, $M({L:.2f}) = {mL:.3f}$ кН·м")

        blocks.append({"markdown": md, "latex": []})

    return blocks


def _fmt_num(v: float) -> str:
    return f"{float(v):.3f}"


def _support_type_label(s_type: str) -> str:
    labels = {
        "fixed": "Fixed (жесткая заделка)",
        "hinged": "Hinged (шарнирно-неподвижная)",
        "roller": "Roller (шарнирно-подвижная)",
    }
    return labels.get(s_type, s_type)


def count_independent_cycles_in_bar_graph(nodes: list[Node], elements: list[MemberElement]) -> int:
    """
    Количество независимых замкнутых контуров K по графу стержней (узлы — имена узлов,
    рёбра — стержни). Для каждой связной компоненты: K_c = max(0, E_c - V_c + 1)
    (цикломатическое число; параллельные стержни между одной парой узлов учитываются как отдельные рёбра).

    Полная классическая оценка степени статической неопределимости рам с жёсткими узлами
    и произвольной топологией требует учёта шарнирных узлов, составных шарниров и т.д.;
    здесь K — только топологический цикл по «жёсткой» схеме стержней из редактора.
    """
    if not elements:
        return 0
    node_names = {str(n.name) for n in nodes}
    adj: dict[str, set[str]] = {}
    edges: list[tuple[str, str]] = []
    for el in elements:
        a, b = _element_endpoint_names(el, nodes)
        if not a or not b or a not in node_names or b not in node_names or a == b:
            continue
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
        edges.append((a, b))
    if not edges:
        return 0

    visited: set[str] = set()
    total_k = 0
    for seed in list(adj.keys()):
        if seed in visited:
            continue
        stack = [seed]
        comp: set[str] = set()
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp.add(u)
            for v in adj.get(u, ()):
                if v not in visited:
                    stack.append(v)
        e_c = sum(1 for a, b in edges if a in comp and b in comp)
        v_c = len(comp)
        total_k += max(0, e_c - v_c + 1)
    return total_k


def calculate_kinematic_analysis(
    nodes: list[Node],
    elements: list[MemberElement],
    supports: list[SupportEntry],
    hinges: list[HingeEntry],
) -> KinematicAnalysisResult:
    """
    Кинематический анализ по формуле Чебышёва (строительная механика):
      W = 3·D − 2·Ш − C₀
    где D — число жёстких дисков, Ш — число простых шарниров, C₀ — число опорных связей.
    """
    # C0: опорные связи
    c0 = 0
    for s in supports:
        s_type = s.get("type")
        if s_type == "roller":
            c0 += 1
        elif s_type == "hinged":
            c0 += 2
        elif s_type == "fixed":
            c0 += 3

    if not elements:
        w0 = 3 * 0 - 2 * 0 - c0
        return {"W": w0, "D": 0, "Sh": 0, "C0": c0}

    # Подготовка: флаги внутренних шарниров на концах стержней.
    hinge_flags: set[tuple[UUID, str]] = set()
    for h in hinges:
        eid = coerce_element_uuid(h.get("element_id"))
        pos = h.get("position")
        if eid is not None and pos in ("start", "end"):
            hinge_flags.add((eid, str(pos)))

    # Узел -> список (элемент, этот конец шарнирный?)
    node_endpoints: dict[str, list[tuple[UUID, bool]]] = {}
    for el in elements:
        na, nb = _element_endpoint_names(el, nodes)
        if not na or not nb:
            continue
        # Tie-rod behaves as pinned at both ends for kinematic counting.
        is_start_hinged = bool(getattr(el, "is_tie", False)) or (el.id, "start") in hinge_flags
        is_end_hinged = bool(getattr(el, "is_tie", False)) or (el.id, "end") in hinge_flags
        node_endpoints.setdefault(na, []).append((el.id, is_start_hinged))
        node_endpoints.setdefault(nb, []).append((el.id, is_end_hinged))

    # Union–find по стержням: объединяем элементы, жёстко связанные в узлах без шарниров.
    parent: dict[UUID, UUID] = {}

    def find(x: UUID) -> UUID:
        root = parent.get(x, x)
        while parent.get(root, root) != root:
            root = parent[root]
        # path compression
        cur = x
        while parent.get(cur, cur) != root:
            nxt = parent[cur]
            parent[cur] = root
            cur = nxt
        parent[x] = root
        return root

    def union(a: UUID, b: UUID) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for el in elements:
        parent.setdefault(el.id, el.id)

    # В каждом узле без шарниров соединяем все примыкающие стержни в один диск.
    for _node_name, endpoints in node_endpoints.items():
        rigid_elements = [eid for (eid, is_hinged) in endpoints if not is_hinged]
        if len(rigid_elements) < 2:
            continue
        base = rigid_elements[0]
        for eid in rigid_elements[1:]:
            union(base, eid)

    # Количество жёстких дисков D по стержням.
    disk_roots: set[UUID] = {find(el.id) for el in elements}
    d_disks = len(disk_roots)

    # Шарниры: для каждого узла с хотя бы одним шарниром считаем,
    # сколько различных дисков в нём сходится; вклад = max(0, k - 1).
    hinge_nodes: set[str] = set()
    element_by_id = {e.id: e for e in elements}
    for h in hinges:
        eid = coerce_element_uuid(h.get("element_id"))
        pos = h.get("position")
        el = element_by_id.get(eid) if eid is not None else None
        if not el or pos not in ("start", "end"):
            continue
        na, nb = _element_endpoint_names(el, nodes)
        node_name = na if pos == "start" else nb
        if node_name:
            hinge_nodes.add(node_name)

    # Ш: базовый вклад от "явных" шарнирных узлов (кратность k − 1).
    sh_simple = 0
    for node_name in hinge_nodes:
        endpoints = node_endpoints.get(node_name, [])
        if not endpoints:
            continue
        hinged_disks: set[UUID] = set()
        for eid, is_hinged in endpoints:
            if is_hinged:
                hinged_disks.add(find(eid))
        k = len(hinged_disks)
        if k > 1:
            sh_simple += k - 1

    # Для затяжек (is_tie=True) шарниры должны считаться принудительно:
    # по условию каждая затяжка добавляет 2 простых шарнира (по одному на каждом конце).
    tie_count = sum(1 for el in elements if bool(getattr(el, "is_tie", False)))
    sh_simple += 2 * tie_count

    w = 3 * d_disks - 2 * sh_simple - c0
    return {"W": w, "D": d_disks, "Sh": sh_simple, "C0": c0}


def calculate_static_indeterminacy_n(
    nodes: list[Node],
    elements: list[MemberElement],
    supports: list[SupportEntry],
    hinges: list[HingeEntry],
) -> tuple[int, int, SupportCounts, int]:
    """
    Степень статической неопределимости n.

    Для схем без замкнутых контуров по графу стержней (K = 0) используется упрощённая
    модель единого жёсткого диска:
      n = Sop - 3 - hinge_relief,
    где hinge_relief — «лишние» шарниры в узле (для k шарнирных концов в одном узле снимается (k-1)).

    При K >= 1 применяется распространённая для плоских рам формула вида:
      n = 3*K - Sh - Sop,
    где Sh — число заданных в редакторе простых шарниров (записей в hinges), Sop — число
    опорных связей. Это ориентировочная оценка для сложных контурных схем; при сомнениях
    остаётся расчёт МКЭ.

    Если бы вычисление K по графу было ненадёжным (нет рёбер, несогласованные узлы),
    можно было бы оставить только упрощённую формулу — здесь K считается однозначно
    по списку стержней, а при K = 0 сохраняется прежнее поведение интерфейса.
    """
    support_counts: SupportCounts = {"fixed": 0, "hinged": 0, "roller": 0}
    for support in supports:
        s_type = support.get("type")
        if s_type in support_counts:
            support_counts[s_type] += 1

    sop = 3 * support_counts["fixed"] + 2 * support_counts["hinged"] + 1 * support_counts["roller"]
    # Internal hinges reduce static indeterminacy.
    # For each node with k hinged member ends, reduction = (k - 1).
    node_hinge_counts: dict[str, int] = {}
    element_by_id = {e.id: e for e in elements}
    for h in hinges:
        eid = coerce_element_uuid(h.get("element_id"))
        pos = h.get("position")
        el = element_by_id.get(eid) if eid is not None else None
        if not el:
            continue
        na, nb = _element_endpoint_names(el, nodes)
        if pos == "start":
            node_name = na
        elif pos == "end":
            node_name = nb
        else:
            continue
        node_hinge_counts[node_name] = node_hinge_counts.get(node_name, 0) + 1

    hinge_relief = sum(max(0, k - 1) for k in node_hinge_counts.values())
    n_simple = sop - 3 - hinge_relief

    k_loops = count_independent_cycles_in_bar_graph(nodes, elements)
    sh_total = len(hinges)
    if k_loops > 0:
        n_val = 3 * k_loops - sh_total - sop
    else:
        n_val = n_simple

    return n_val, sop, support_counts, k_loops


def _eq_latex_num(v: float) -> str:
    """Компактное число для LaTeX (кН, кН·м, м)."""
    x = float(v)
    if abs(x) < 1e-14:
        return "0"
    s = f"{x:.5g}"
    if "e" in s.lower():
        s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s


def build_equilibrium_report(
    ss_load: SystemElements,
    nodes: list[Node],
    elements: list[MemberElement],
    supports: list[SupportEntry],
    loads: list[LoadEntry],
    hinges: list[HingeEntry],
) -> list[str]:
    """
    LaTeX-строки глобальных уравнений равновесия плоской системы от внешних нагрузок и реакций опор.
    Для n ≠ 0 — краткое сообщение про МКЭ (рукописная развёртка не формируется).
    """
    n_val, _, _, _ = calculate_static_indeterminacy_n(nodes, elements, supports, hinges)
    if n_val != 0:
        return [r"\text{Реакции определены методом конечных элементов (МКЭ).}"]

    if not supports:
        return [r"\text{Нет опор — уравнения равновесия не формулируются.}"]

    node_xy = {str(n.name): (float(n.x), float(n.y)) for n in nodes}
    elem_by_id = {e.id: e for e in elements}

    ref = str(supports[0]["node"])
    if ref not in node_xy:
        return [r"\text{Не удалось сопоставить первую опору с узлом для уравнения моментов.}"]
    xa, ya = node_xy[ref]

    ext_fx = 0.0
    ext_fy = 0.0
    ext_m_a = 0.0

    for load in loads:
        if load["type"] == "point":
            nd = str(load.get("node", ""))
            if nd not in node_xy:
                continue
            xp, yp = node_xy[nd]
            fx = float(load.get("Fx", 0.0))
            fy = float(load.get("Fy", 0.0))
            ext_fx += fx
            ext_fy += fy
            ext_m_a += (xp - xa) * fy - (yp - ya) * fx
        elif load["type"] == "moment":
            ext_m_a += float(load.get("M", 0.0))
        elif load["type"] == "distributed":
            leu = coerce_element_uuid(load.get("element_id"))
            el = elem_by_id.get(leu) if leu is not None else None
            if not el:
                continue
            na, nb = _element_endpoint_names(el, nodes)
            if na not in node_xy or nb not in node_xy:
                continue
            x1, y1 = node_xy[na]
            x2, y2 = node_xy[nb]
            dx, dy = x2 - x1, y2 - y1
            L = float(np.hypot(dx, dy))
            if L < 1e-12:
                continue
            tcx, tcy = dx / L, dy / L
            nx, ny = -tcy, tcx
            q = float(load.get("q", 0.0))
            fqx, fqy = q * L * nx, q * L * ny
            xm, ym = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            ext_fx += fqx
            ext_fy += fqy
            ext_m_a += (xm - xa) * fqy - (ym - ya) * fqx

    reaction_rows, _miss = collect_support_reaction_rows(ss_load, nodes, supports)
    if not reaction_rows:
        return [r"\text{Нет данных реакций для уравнений равновесия.}"]

    sum_rx = sum(float(r["Fx"]) for r in reaction_rows)
    sum_ry = sum(float(r["Fy"]) for r in reaction_rows)

    m_reac = 0.0
    for r in reaction_rows:
        nm = str(r["Узел"])
        if nm not in node_xy:
            continue
        xb, yb = node_xy[nm]
        Rbx = float(r["Fx"])
        Rby = float(r["Fy"])
        Mb = float(r["Mz (Tz)"])
        m_reac += (xb - xa) * Rby - (yb - ya) * Rbx + Mb

    rx_join = " + ".join("R_{" + str(r["Узел"]) + "x}" for r in reaction_rows)
    ry_join = " + ".join("R_{" + str(r["Узел"]) + "y}" for r in reaction_rows)

    line_intro = (
        rf"\text{{Уравнения равновесия (центр моментов — узел }}{ref}\text{{, силы в кН, длины в м):}}"
    )

    line_mx = (
        rf"\Sigma M_{{{ref}}} = 0:\quad M^{{\text{{внеш}}}} + M^{{\text{{реак}}}} = 0 \Rightarrow "
        rf"{_eq_latex_num(ext_m_a)} + ({_eq_latex_num(m_reac)}) = {_eq_latex_num(ext_m_a + m_reac)} "
        rf"\approx 0\ \text{{кН·м}}"
    )

    line_x = (
        rf"\Sigma X = 0:\quad {rx_join} + F_x^{{\text{{н}}}} = 0 \Rightarrow "
        rf"{_eq_latex_num(sum_rx)} + ({_eq_latex_num(ext_fx)}) = {_eq_latex_num(sum_rx + ext_fx)} "
        rf"\approx 0\ \text{{кН}}"
    )

    line_y = (
        rf"\Sigma Y = 0:\quad {ry_join} + F_y^{{\text{{н}}}} = 0 \Rightarrow "
        rf"{_eq_latex_num(sum_ry)} + ({_eq_latex_num(ext_fy)}) = {_eq_latex_num(sum_ry + ext_fy)} "
        rf"\approx 0\ \text{{кН}}"
    )

    return [line_intro, line_mx, line_x, line_y]


def calculate_unit_displacement(
    base_ss: SystemElements,
    nodes: list[Node],
    node_name: str,
    unit_direction: UnitDirection,
) -> tuple[SystemElements | None, NodeDisplacement | None, str]:
    """
    Unit-state analysis (Mohr-method equivalent via FE displacement extraction).
    Returns ``(ss_unit, disp, error_message)``; on solver failure ``ss_unit`` and ``disp`` are ``None``.
    """
    name_to_id = node_name_to_model_node_id(base_ss, nodes)
    if node_name not in name_to_id:
        return None, None, "Выбранный узел не найден в расчетной модели."

    node_id = name_to_id[node_name]
    ss_unit = copy.deepcopy(base_ss)
    ss_unit.remove_loads()

    if unit_direction == "x":
        ss_unit.point_load(node_id=node_id, Fx=1.0, Fy=0.0)
    elif unit_direction == "y":
        ss_unit.point_load(node_id=node_id, Fx=0.0, Fy=-1.0)
    elif unit_direction == "rz":
        ss_unit.moment_load(node_id=node_id, Tz=1.0)
    else:
        return None, None, "Неизвестное направление единичной нагрузки."

    ok, err = try_solve(ss_unit)
    if not ok:
        return None, None, err
    disp = cast(NodeDisplacement, ss_unit.get_node_displacements(node_id))
    return ss_unit, disp, ""


def _user_sym_to_latex(sym: str) -> str:
    """Turn user label like c_Ax or φ_A into LaTeX math (first underscore → subscript)."""
    s = (sym or "?").strip() or "?"
    if "_" in s:
        base, sub = s.split("_", 1)
        if base in ("φ", "phi", "\u03c6"):
            return rf"\varphi_{{{sub}}}"
        return rf"{base}_{{{sub}}}"
    return s


def _disp_delta_m_mm_latex(delta: float, sub_text: str) -> str:
    """Учебная строка: Δ_sub = … м = … мм (только отображение)."""
    mm = float(delta) * 1000.0
    return rf"\Delta_{{\text{{{sub_text}}}}} = {delta:.4f}\,\text{{м}} = {mm:.1f}\,\text{{мм}}"


def _disp_delta_total_p_c_latex(delta_p: float, delta_c: float, delta_tot: float) -> str:
    """Итог: Δ = Δ_P + Δ_c = … м = … мм (только отображение)."""
    dtot = float(delta_tot)
    mm = dtot * 1000.0
    return (
        rf"\Delta = \Delta_P + \Delta_c = {float(delta_p):.4f}\,\text{{м}} + ({float(delta_c):.4f}\,\text{{м}}) = "
        rf"{dtot:.4f}\,\text{{м}} = {mm:.1f}\,\text{{мм}}"
    )


def _default_displacement_entry_from_support(support: SupportEntry) -> SupportDisplacementEntry:
    """Нулевая запись осадок для опоры (совместимо с ``compute_settlement_component``)."""
    node = str(support["node"])
    stype = support["type"]
    if stype == "roller":
        return {
            "dn_mm": 0.0,
            "dx_mm": 0.0,
            "dy_mm": 0.0,
            "phi_rad": 0.0,
            "sym_dn": f"c_{node}",
        }
    if stype == "hinged":
        return {
            "dx_mm": 0.0,
            "dy_mm": 0.0,
            "phi_rad": 0.0,
            "sym_dx": f"c_{node}x",
            "sym_dy": f"c_{node}y",
        }
    return {
        "dx_mm": 0.0,
        "dy_mm": 0.0,
        "phi_rad": 0.0,
        "sym_dx": f"c_{node}x",
        "sym_dy": f"c_{node}y",
        "sym_phi": f"φ_{node}",
    }


def support_settlement_rows_to_map(
    rows: Iterable[Mapping[str, Any]],
    supports: list[SupportEntry],
) -> SupportDisplacementsMap:
    """
    Сводит табличные осадки (м по X,Y,N; рад по Rot) к карте ``SupportDisplacementsMap``
    для ``compute_settlement_component`` / метода сил.
    """
    out: SupportDisplacementsMap = {}
    for s in supports:
        sid = int(s["id"])
        out[sid] = dict(_default_displacement_entry_from_support(s))  # type: ignore[arg-type]
    for row in rows:
        try:
            sid = int(row.get("support_id", -1))
        except (TypeError, ValueError):
            continue
        if sid not in out:
            continue
        direction = str(row.get("direction", "")).strip()
        sym = str(row.get("symbol", "")).strip()
        try:
            val = float(row.get("value", 0.0))
        except (TypeError, ValueError):
            val = 0.0
        base = out[sid]
        s_up = next((x for x in supports if int(x["id"]) == sid), None)
        stype = str(s_up["type"]) if s_up else "hinged"

        if direction == "X":
            base["dx_mm"] = val * 1000.0
            if sym:
                base["sym_dx"] = sym
        elif direction == "Y":
            base["dy_mm"] = val * 1000.0
            if sym:
                base["sym_dy"] = sym
        elif direction == "Rot":
            if stype == "fixed":
                base["phi_rad"] = val
                if sym:
                    base["sym_phi"] = sym
        elif direction == "N":
            if stype == "roller":
                base["dn_mm"] = val * 1000.0
                if sym:
                    base["sym_dn"] = sym
        out[sid] = base
    return out


def compute_settlement_component(
    ss_unit: SystemElements,
    nodes: list[Node],
    supports: list[SupportEntry],
    support_displacements: SupportDisplacementsMap,
) -> tuple[float, str, str, list[str], list[str], str, list[str], list[str], str, str, str]:
    """
    Compute settlement contribution:
      Δ_c = - Σ (R_i^(1) · Δ_i)
    where R_i^(1) are support reactions from the unit (virtual) state and Δ_i are prescribed settlements.

    Returns:
        delta_c, settlement_latex, settlement_text, warnings, settlement_steps_latex
        (пустой список — пошаговый LaTeX не формируется),
        settlement_formula_latex, reaction_summary_lines (опоры с ненулевыми осадками),
        unit_reactions_latex (реакции всех опор в единичном состоянии),
        compact_settlement_latex (одна строка: символы, числа в скобках, итог в м и мм),
        symbolic_inner_sum (сумма слагаемых для Δ_is в отчёте метода сил),
        numeric_inner_sum (численная сумма в тех же скобках).
    """
    unit_reactions_latex, unit_rx_missed = compute_unit_reactions_for_settlement(
        ss_unit, nodes, supports, support_displacements
    )
    warnings: list[str] = list(unit_rx_missed)

    name_to_id = node_name_to_model_node_id(ss_unit, nodes)

    term_values: list[float] = []
    term_outer_parts: list[str] = []
    term_text_parts: list[str] = []
    term_numeric_parts: list[str] = []
    settlement_steps_latex: list[str] = []
    reaction_summary_lines: list[str] = []

    eps_mm = 1e-9
    eps_rad = 1e-12

    def support_has_settlement_input(support: SupportEntry, disp: SupportDisplacementEntry) -> bool:
        stype = support["type"]
        if stype == "roller":
            return abs(float(disp.get("dn_mm", 0.0))) > eps_mm
        if stype == "hinged":
            return abs(float(disp.get("dx_mm", 0.0))) > eps_mm or abs(float(disp.get("dy_mm", 0.0))) > eps_mm
        return (
            abs(float(disp.get("dx_mm", 0.0))) > eps_mm
            or abs(float(disp.get("dy_mm", 0.0))) > eps_mm
            or abs(float(disp.get("phi_rad", 0.0))) > eps_rad
        )

    for support in supports:
        sid = int(support["id"])
        node_name = support["node"]
        node_id = name_to_id.get(node_name)
        if node_id is None:
            warnings.append(f"S{sid}: узел '{node_name}' не найден в единичной схеме.")
            continue

        disp = support_displacements.get(sid, {})
        dx_m = float(disp.get("dx_mm", 0.0)) / 1000.0
        dy_m = float(disp.get("dy_mm", 0.0)) / 1000.0
        phi = float(disp.get("phi_rad", 0.0))
        s_type = support["type"]

        node = ss_unit.node_map[node_id]
        rx = float(node.Fx)
        ry = float(node.Fy)
        rm = float(node.Tz)

        if s_type == "roller":
            dn_m = float(disp.get("dn_mm", 0.0)) / 1000.0
            if abs(dn_m) <= 1e-15:
                continue
            angle_deg = float(support.get("angle", 90.0))
            a = np.radians(angle_deg)
            R_eq = rx * float(np.cos(a)) + ry * float(np.sin(a))
            val = R_eq * dn_m
            term_values.append(val)
            term_outer_parts.append(rf"R_{{{node_name}n}}\cdot\Delta_{{{node_name}n}}")
            term_text_parts.append(f"{R_eq:.6g}·{dn_m:.6g}")
            term_numeric_parts.append(f"{val:.4g}")
            continue

        if abs(dx_m) > 1e-15:
            val = rx * dx_m
            term_values.append(val)
            Rbx = rf"R_{{{node_name}x}}"
            term_outer_parts.append(rf"{Rbx}\cdot\Delta_{{{node_name}x}}")
            term_text_parts.append(f"{rx:.6g}·{dx_m:.6g}")
            term_numeric_parts.append(f"{val:.4g}")
        if abs(dy_m) > 1e-15:
            val = ry * dy_m
            term_values.append(val)
            Rby = rf"R_{{{node_name}y}}"
            term_outer_parts.append(rf"{Rby}\cdot\Delta_{{{node_name}y}}")
            term_text_parts.append(f"{ry:.6g}·{dy_m:.6g}")
            term_numeric_parts.append(f"{val:.4g}")
        if s_type == "fixed" and abs(phi) > 1e-15:
            val = rm * phi
            term_values.append(val)
            Mb = rf"M_{{{node_name}}}"
            term_outer_parts.append(rf"{Mb}\cdot\varphi_{{{node_name}}}")
            term_text_parts.append(f"{rm:.6g}·{phi:.6g}")
            term_numeric_parts.append(f"{val:.4g}")

    for support in supports:
        sid = int(support["id"])
        node_name = support["node"]
        node_id = name_to_id.get(node_name)
        if node_id is None:
            continue
        disp = support_displacements.get(sid, {})
        if not support_has_settlement_input(support, disp):
            continue
        node = ss_unit.node_map[node_id]
        rx = float(node.Fx)
        ry = float(node.Fy)
        rm = float(node.Tz)
        line = (
            rf"\text{{Узел {node_name}:}}\quad "
            rf"\overline{{R}}_{{{node_name}x}}={rx:.3f}\ \text{{кН}},\ "
            rf"\overline{{R}}_{{{node_name}y}}={ry:.3f}\ \text{{кН}}"
        )
        if support["type"] == "fixed":
            line += rf",\ \overline{{M}}_{{{node_name}}}={rm:.3f}\ \text{{кН·м}}"
        reaction_summary_lines.append(line)

    if not term_values:
        compact_zero = r"\Delta_c = 0\quad\text{(осадки опор не заданы)}"
        return (
            0.0,
            r"\Delta_{c} = 0",
            "0",
            warnings,
            [],
            r"\Delta_{c}=0",
            reaction_summary_lines,
            unit_reactions_latex,
            compact_zero,
            "",
            "",
        )

    s = sum(term_values)
    delta_c = -s
    sym_outer = " + ".join(term_outer_parts)
    num_inner = " + ".join(term_numeric_parts)
    latex_expr = r"\Delta_{c} = -\left(" + sym_outer + r"\right)"
    latex_formula = (
        r"\Delta_{c} = -\left("
        + sym_outer
        + r"\right) = -\left("
        + num_inner
        + r"\right) = "
        + f"{delta_c:.4f}"
    )
    text_expr = "-(" + " + ".join(term_text_parts) + f") = {delta_c:.8g}"
    compact_settlement_latex = (
        rf"\Delta_c = -\left({sym_outer}\right) = -\left({num_inner}\right) = {delta_c:.4f}\,\text{{м}} = "
        rf"{delta_c * 1000.0:.1f}\,\text{{мм}}"
    )
    return (
        delta_c,
        latex_expr,
        text_expr,
        warnings,
        settlement_steps_latex,
        latex_formula,
        reaction_summary_lines,
        unit_reactions_latex,
        compact_settlement_latex,
        sym_outer,
        num_inner,
    )


def _displacement_component_key(unit_direction: UnitDirection) -> str:
    return {"x": "ux", "y": "uy", "rz": "phi_z"}[unit_direction]


def _format_component_name(unit_direction: UnitDirection) -> str:
    return {"x": "Δx", "y": "Δy", "rz": "φ"}[unit_direction]


def _format_component_value(unit_direction: UnitDirection, value: float) -> str:
    if unit_direction in ("x", "y"):
        return f"{value * 1000:.3f} мм"
    return f"{value:.6f} рад"


def compute_mohr_displacement_report(
    ss_load: SystemElements, ss_unit: SystemElements
) -> tuple[float, list[MohrIntegralRow], list[str]]:
    """
    Compute displacement using Mohr integral:
      delta = sum( integral(M * m dx) / EI )
    using numerical integration along each element.
    """
    rows: list[MohrIntegralRow] = []
    warnings: list[str] = []
    total = 0.0

    common_ids = sorted(set(ss_load.element_map.keys()) & set(ss_unit.element_map.keys()))
    if not common_ids:
        return 0.0, rows, ["Не найдены общие элементы между грузовым и единичным состояниями."]

    for el_id in common_ids:
        e_load = ss_load.element_map[el_id]
        e_unit = ss_unit.element_map[el_id]

        m_load = np.asarray(getattr(e_load, "bending_moment", None), dtype=float)
        m_unit = np.asarray(getattr(e_unit, "bending_moment", None), dtype=float)
        length = float(getattr(e_load, "l", 0.0) or 0.0)
        ei = float(getattr(e_load, "EI", 0.0) or 0.0)

        if length <= 0:
            warnings.append(f"E{el_id}: длина элемента некорректна, элемент пропущен.")
            continue
        if ei == 0:
            warnings.append(f"E{el_id}: EI = 0 или не задан, элемент пропущен.")
            continue
        if m_load.size == 0 or m_unit.size == 0:
            warnings.append(f"E{el_id}: отсутствуют эпюры моментов, элемент пропущен.")
            continue

        n = min(m_load.size, m_unit.size)
        m_load = m_load[:n]
        m_unit = m_unit[:n]
        x = np.linspace(0.0, length, n)

        area_m = float(np.trapz(m_load, x))
        area_abs_m = float(np.trapz(np.abs(m_load), x))
        if area_abs_m > 1e-12:
            m_unit_weighted = float(np.trapz(np.abs(m_load) * m_unit, x) / area_abs_m)
        else:
            m_unit_weighted = float(np.mean(m_unit))

        integral_mm = float(np.trapz(m_load * m_unit, x))
        contribution = integral_mm / ei
        total += contribution

        rows.append(
            {
                "Элемент": f"E{el_id}",
                "L, м": round(length, 4),
                "EI": round(ei, 6),
                "Площадь Mгр, кН·м²": round(area_m, 6),
                "Ордината Mед (взв.)": round(m_unit_weighted, 6),
                "∫(Mгр*Mед)dx": round(integral_mm, 6),
                "Вклад ∫/EI": round(contribution, 9),
            }
        )

    return total, rows, warnings


def _detect_epure_type(m0: float, mm: float, m1: float) -> str:
    """Heuristic epure type for textual report."""
    eps = 1e-9
    if abs(m0) < eps and abs(mm) < eps and abs(m1) < eps:
        return "нулевая"
    if abs(m0 - m1) < eps and abs(mm - m0) < eps:
        return "прямоугольная"
    linear_mid = 0.5 * (m0 + m1)
    if abs(mm - linear_mid) <= max(1e-6, 0.03 * (abs(m0) + abs(m1) + abs(mm) + 1e-9)):
        if abs(m0) < eps or abs(m1) < eps:
            return "треугольная"
        return "трапециевидная"
    return "параболическая/криволинейная"


def build_vereshchagin_report(
    ss_load: SystemElements,
    ss_unit: SystemElements,
    session_to_ana: dict[UUID, int],
    nodes: list[Node],
    elements: list[MemberElement],
) -> VereshchaginReport:
    ana_to_session = {int(v): k for k, v in session_to_ana.items()}
    node_xy = _nodes_name_map(nodes)

    def n(v: float) -> str:
        s = f"{float(v):.4f}".rstrip("0").rstrip(".")
        return s if s else "0"

    def _close_xy(a: tuple[float, float], b: tuple[float, float], tol: float = 1e-5) -> bool:
        return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol

    def section_label(element_id: int, e_load: Any) -> str:
        sess_uuid = ana_to_session.get(int(element_id))
        element = None
        if sess_uuid is not None:
            element = next((e for e in elements if e.id == sess_uuid), None)
        if element:
            a, b = _element_endpoint_names(element, nodes)
            return f"{a}{b}"
        x1, y1 = float(e_load.vertex_1.x), float(e_load.vertex_1.y)
        x2, y2 = float(e_load.vertex_2.x), float(e_load.vertex_2.y)
        for se in elements:
            na, nb = _element_endpoint_names(se, nodes)
            nda = node_xy.get(na)
            ndb = node_xy.get(nb)
            if not nda or not ndb:
                continue
            p1 = (float(nda.x), float(nda.y))
            p2 = (float(ndb.x), float(ndb.y))
            if (_close_xy((x1, y1), p1) and _close_xy((x2, y2), p2)) or (_close_xy((x1, y1), p2) and _close_xy((x2, y2), p1)):
                return f"{na}{nb}"
        return f"E{element_id}"

    lines_latex: list[str] = []
    lines_text: list[str] = []
    warnings: list[str] = []
    total_delta = 0.0
    expr_terms_common_ei: list[str] = []
    expr_terms_var_ei: list[str] = []
    omega_y_terms: list[str] = []
    omega_y_symbolic_parts: list[str] = []
    omega_y_sum = 0.0
    ei_values: list[float] = []
    k_global = 0

    common_ids = sorted(set(ss_load.element_map.keys()) & set(ss_unit.element_map.keys()))
    for el_id in common_ids:
        e_load = ss_load.element_map[el_id]
        e_unit = ss_unit.element_map[el_id]
        m_load = np.asarray(getattr(e_load, "bending_moment", None), dtype=float)
        m_unit = np.asarray(getattr(e_unit, "bending_moment", None), dtype=float)
        length = float(getattr(e_load, "l", 0.0) or 0.0)
        ei = float(getattr(e_load, "EI", 0.0) or 0.0)
        label = section_label(el_id, e_load)

        if length <= 0:
            warnings.append(f"{label}: некорректная длина, пропуск.")
            continue
        if ei == 0:
            warnings.append(f"{label}: EI=0/не задан, пропуск.")
            continue
        if m_load.size == 0 or m_unit.size == 0:
            warnings.append(f"{label}: отсутствуют эпюры M, пропуск.")
            continue

        n_pts = min(m_load.size, m_unit.size)
        m_load = m_load[:n_pts]
        m_unit = m_unit[:n_pts]
        x = np.linspace(0.0, length, n_pts)

        m0 = float(m_load[0])
        mm = float(m_load[len(m_load) // 2])
        m1 = float(m_load[-1])
        linear_mid = 0.5 * (m0 + m1)
        is_linear = abs(mm - linear_mid) <= max(1e-6, 0.03 * (abs(m0) + abs(m1) + abs(mm) + 1e-9))

        section_terms: list[tuple[str, float, float, float, str]] = []
        # (omega_formula, omega_value, xcg, y_value, term_expr)

        def add_term(omega_formula: str, omega_value: float, xcg: float) -> None:
            if abs(omega_value) < 1e-12:
                return
            y_val = float(np.interp(xcg, x, m_unit))
            omega_latex = omega_formula.replace("*", r"\cdot ")
            term_expr = rf"({omega_latex})\cdot{n(y_val)}"
            section_terms.append((omega_formula, omega_value, xcg, y_val, term_expr))

        if is_linear:
            # linear: triangle / rectangle / trapezoid decomposition
            if abs(m0) < 1e-12 and abs(m1) < 1e-12:
                continue
            if abs(m0 - m1) < 1e-12:
                omega_formula = f"{n(length)}*{n(m0)}"
                add_term(omega_formula, length * m0, length / 2)
            elif m0 * m1 >= 0:
                if abs(m1) >= abs(m0):
                    # rectangle at start value + triangle to end
                    omega_rect_f = f"{n(length)}*{n(m0)}"
                    add_term(omega_rect_f, length * m0, length / 2)
                    h = m1 - m0
                    omega_tri_f = f"1/2*{n(length)}*{n(h)}"
                    add_term(omega_tri_f, 0.5 * length * h, 2 * length / 3)
                else:
                    omega_rect_f = f"{n(length)}*{n(m1)}"
                    add_term(omega_rect_f, length * m1, length / 2)
                    h = m0 - m1
                    omega_tri_f = f"1/2*{n(length)}*{n(h)}"
                    add_term(omega_tri_f, 0.5 * length * h, length / 3)
            else:
                # opposite signs: split into 2 triangles by zero crossing
                x0 = length * abs(m0) / (abs(m0) + abs(m1))
                omega1_f = f"1/2*{n(x0)}*{n(m0)}"
                add_term(omega1_f, 0.5 * x0 * m0, x0 / 3)
                l2 = length - x0
                omega2_f = f"1/2*{n(l2)}*{n(m1)}"
                add_term(omega2_f, 0.5 * l2 * m1, x0 + 2 * l2 / 3)
        else:
            # nonlinear: linear chord + parabolic segment (Vereshchagin style)
            # linear chord as trapezoid (single term)
            omega_lin = 0.5 * length * (m0 + m1)
            omega_lin_f = f"1/2*{n(length)}*({n(m0)}+{n(m1)})"
            add_term(omega_lin_f, omega_lin, length / 2)

            f_par = mm - linear_mid
            omega_par = (2.0 / 3.0) * length * f_par
            omega_par_f = f"2/3*{n(length)}*{n(f_par)}"
            add_term(omega_par_f, omega_par, length / 2)

        if not section_terms:
            continue

        section_chunks: list[str] = []
        section_sum_product = 0.0
        for omega_formula, omega_value, _xcg, y_val, term_expr in section_terms:
            product = omega_value * y_val
            section_sum_product += product
            k_global += 1
            omega_y_symbolic_parts.append(rf"\omega_{{{k_global}}}\cdot y_{{{k_global}}}")
            expr_terms_common_ei.append(term_expr)
            expr_terms_var_ei.append(
                r"\frac{" + n(product) + r"}{\mathrm{EI}_{\text{" + str(label) + r"}}}"
            )
            omega_y_terms.append(n(product))
            omega_y_sum += product
            ei_values.append(ei)
            total_delta += product / ei
            omega_show = omega_formula.replace("*", r"\cdot ")
            section_chunks.append(
                rf"\omega = {omega_show} = {n(omega_value)},\ y = {n(y_val)} \Rightarrow \omega\cdot y = {n(product)}"
            )

        joined = r";\ ".join(section_chunks)
        lines_latex.append(
            rf"\text{{Участок {label} }}\ ({n(length)}\,\text{{м}}):\quad {joined}"
        )
        lines_text.append(f"Участок {label} ({n(length)} м): " + joined.replace(r"\Rightarrow", "=>"))

    ei_uniform: float | None = None
    if not expr_terms_common_ei:
        final_formula = r"\Delta_{P} = 0"
        compact_formula = r"\Delta_{P} = 0"
        same_ei = True
        omega_y_sum_out = 0.0
    else:
        same_ei = max(ei_values) - min(ei_values) <= 1e-9 if ei_values else True
        sym_wy = " + ".join(omega_y_symbolic_parts)
        if same_ei:
            final_formula = rf"\Delta_{{P}} = \frac{{({sym_wy})}}{{\mathrm{{EI}}}}"
            compact_formula = rf"\Delta_{{P}} = \frac{{{n(omega_y_sum)}}}{{\mathrm{{EI}}}}"
            ei_uniform = float(ei_values[0]) if ei_values else None
        else:
            final_formula = r"\Delta_{P} = " + " + ".join(expr_terms_var_ei)
            compact_formula = final_formula
        omega_y_sum_out = float(omega_y_sum)

    full_text = "\n".join(lines_text + [final_formula, compact_formula])
    return (
        lines_latex,
        final_formula,
        compact_formula,
        total_delta,
        warnings,
        full_text,
        same_ei,
        omega_y_sum_out,
        ei_uniform,
    )


def build_unit_preview_model(
    base_ss: SystemElements, nodes: list[Node], node_name: str, unit_direction: UnitDirection
) -> SystemElements:
    """Create temporary model for preview of a single unit action."""
    name_to_id = node_name_to_model_node_id(base_ss, nodes)
    if node_name not in name_to_id:
        raise ValueError("Выбранный узел не найден в расчетной модели.")

    node_id = name_to_id[node_name]
    ss_preview = copy.deepcopy(base_ss)
    ss_preview.remove_loads()

    if unit_direction == "x":
        ss_preview.point_load(node_id=node_id, Fx=1.0, Fy=0.0)
    elif unit_direction == "y":
        ss_preview.point_load(node_id=node_id, Fx=0.0, Fy=-1.0)
    elif unit_direction == "rz":
        ss_preview.moment_load(node_id=node_id, Tz=1.0)
    else:
        raise ValueError("Неизвестное направление единичной нагрузки.")

    return ss_preview


def _solved_unit_system_for_force_method(
    base_solved: SystemElements, nodes: list[Node], node_name: str, unit_direction: UnitDirection
) -> SystemElements:
    """Модель с единичным воздействием и выполненным solve (эпюры M̄ для метода сил)."""
    ss = build_unit_preview_model(base_solved, nodes, node_name, unit_direction)
    ok, err = try_solve(ss)
    if not ok:
        raise RuntimeError(err)
    return ss


def _fmt_canonical_coef(v: float) -> str:
    x = float(v)
    if abs(x) < 1e-14:
        return "0"
    s = f"{x:.4g}"
    if "e" in s.lower():
        s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def tie_member_elements(elements: list[MemberElement], hinges: list[HingeEntry]) -> list[MemberElement]:
    """Затяжка: стержень, у которого в hinges заданы и start, и end."""
    hinge_ends: dict[UUID, set[str]] = {}
    for h in hinges:
        eid = coerce_element_uuid(h.get("element_id"))
        pos = h.get("position")
        if eid is not None and pos in ("start", "end"):
            hinge_ends.setdefault(eid, set()).add(str(pos))
    ties_by_flag = [e for e in elements if bool(getattr(e, "is_tie", False))]
    ties_by_hinges = [e for e in elements if hinge_ends.get(e.id, set()) >= {"start", "end"}]
    # Priority: explicit is_tie, then implicit by hinges.
    uniq: dict[UUID, MemberElement] = {e.id: e for e in ties_by_flag}
    for e in ties_by_hinges:
        uniq.setdefault(e.id, e)
    return list(uniq.values())


def _hinges_excluding_elements(hinges: list[HingeEntry], excluded: set[UUID]) -> list[HingeEntry]:
    return [h for h in hinges if coerce_element_uuid(h.get("element_id")) not in excluded]


def _member_axis_unit_and_length(el: MemberElement, nodes: list[Node]) -> tuple[float, float, float]:
    """Единичный вектор оси стержня (от start к end) в ГСК и длина L (м)."""
    by = _nodes_uuid_map(nodes)
    n1 = by.get(el.start_node_id)
    n2 = by.get(el.end_node_id)
    if not n1 or not n2:
        return 0.0, 0.0, 0.0
    dx, dy = float(n2.x - n1.x), float(n2.y - n1.y)
    L = float(np.hypot(dx, dy))
    if L < 1e-12:
        return 0.0, 0.0, 0.0
    return dx / L, dy / L, L


def _tie_axial_self_flexibility(el: MemberElement, nodes: list[Node], global_ea: float) -> float:
    """Добавка к δ_ii: N²L/(EA) при N = 1 → L/EA."""
    _ux, _uy, L = _member_axis_unit_and_length(el, nodes)
    if L <= 0.0 or global_ea <= 0.0:
        return 0.0
    return L / float(global_ea)


def _solved_tie_cut_unit_system(
    ss_os: SystemElements,
    nodes: list[Node],
    tie_el: MemberElement,
) -> SystemElements:
    """ОС уже без затяжки; единичное состояние — встречные силы 1 кН вдоль оси удалённого стержня."""
    na, nb = _element_endpoint_names(tie_el, nodes)
    if not na or not nb:
        raise RuntimeError("Не удалось определить узлы затяжки.")
    ux, uy, L = _member_axis_unit_and_length(tie_el, nodes)
    if L < 1e-12:
        raise RuntimeError("Нулевая длина затяжки.")
    name_to_id = node_name_to_model_node_id(ss_os, nodes)
    if na not in name_to_id or nb not in name_to_id:
        raise RuntimeError("Узлы затяжки не найдены в основной системе.")
    ss = copy.deepcopy(ss_os)
    ss.remove_loads()
    ida = name_to_id[na]
    idb = name_to_id[nb]
    ss.point_load(node_id=ida, Fx=float(ux), Fy=float(uy))
    ss.point_load(node_id=idb, Fx=float(-ux), Fy=float(-uy))
    ok, err = try_solve(ss)
    if not ok:
        raise RuntimeError(err)
    return ss


def _force_method_delta_is_vector(
    unit_states: list[SystemElements],
    nodes: list[Node],
    supports: list[SupportEntry],
    sd_map: SupportDisplacementsMap,
    warnings: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Свободные члены от осадок опор: ``Δ_is = -Σ_k (R_ki · c_k)`` для единичного состояния i
    (через ``compute_settlement_component``).
    """
    n = len(unit_states)
    delta_is = np.zeros(n)
    detail: list[str] = []
    for i in range(n):
        comp = compute_settlement_component(unit_states[i], nodes, supports, sd_map)
        delta_is[i] = float(comp[0])
        warnings.extend(comp[3])
        sym_sum, num_sum = comp[9], comp[10]
        i1 = i + 1
        if sym_sum:
            detail.append(
                r"\Delta_{" + str(i1) + r"s} = -\left(" + sym_sum + r"\right) = -\left(" + num_sum + r"\right) = "
                + f"{float(comp[0]):.6g}"
            )
        else:
            detail.append(r"\Delta_{" + str(i1) + r"s} = 0")
    return delta_is, detail


def _append_canonical_latex_with_settlements(
    latex_lines: list[str],
    n: int,
    delta: np.ndarray,
    delta_f: np.ndarray,
    delta_is: np.ndarray,
    delta_is_detail: list[str],
    mohr_numeric_caption: str,
) -> None:
    latex_lines.append(
        rf"\sum_{{j=1}}^{{{n}}} \delta_{{ij}}\, X_j + \Delta_{{iF}} + \Delta_{{is}} = 0,\qquad i=1,\ldots,{n}."
    )
    latex_lines.append(r"\textbf{2. Учёт внешних перемещений (осадок опор)}")
    for ln in delta_is_detail:
        latex_lines.append(ln)
    latex_lines.append(mohr_numeric_caption)
    for i in range(n):
        parts = [rf"{_fmt_canonical_coef(delta[i, j])} \cdot X_{{{j + 1}}}" for j in range(n)]
        parts.append(_fmt_canonical_coef(delta_f[i]))
        parts.append(_fmt_canonical_coef(delta_is[i]))
        latex_lines.append(" + ".join(parts) + r" = 0")
    latex_lines.append(
        r"\text{или в матричном виде:}\quad [\delta]\,\mathbf{X} = -\left(\mathbf{\Delta}_F + \mathbf{\Delta}_s\right)"
    )


def build_force_method_report(
    ss_load_solved: SystemElements,
    nodes: list[Node],
    elements: list[MemberElement],
    supports: list[SupportEntry],
    hinges: list[HingeEntry],
    loads: list[LoadEntry],
    *,
    max_unknowns: int = 10,
    global_ei: float = 5000.0,
    global_ea: float = 1.0e9,
    support_settlement_rows: list[dict[str, Any]] | None = None,
) -> ForceMethodReport:
    """
    Канонические уравнения метода сил: ``Σ δ_ij X_j + Δ_iF + Δ_is = 0``.

    При **W < 0** (Чебышёв) и достаточном числе **затяжек** основная система без этих стержней;
    **δ_ii** дополняется **L/(EA)**. Иначе — единичные силы в узлах на полной модели.

    ``support_settlement_rows`` — строки осадок из сессии (узел, направление, обозначение, величина в м/рад).
    """
    markdown_intro: list[str] = []
    latex_lines: list[str] = []
    markdown_solution: list[str] = []
    warnings: list[str] = []

    kin = calculate_kinematic_analysis(nodes, elements, supports, hinges)
    Wk = int(kin["W"])
    n_target, _sop, _cnt, _k = calculate_static_indeterminacy_n(nodes, elements, supports, hinges)
    ties_all = tie_member_elements(elements, hinges)

    n_need_tie = min(-Wk, int(max_unknowns)) if Wk < 0 else 0
    use_tie_os = bool(Wk < 0 and n_need_tie > 0 and len(ties_all) >= n_need_tie)

    if not use_tie_os and int(n_target) <= 0:
        return {
            "ok": False,
            "markdown_intro": [
                "**Метод сил** в канонической форме строится при **n > 0** (оценка редактора) "
                f"или при **W < 0** по Чебышёву с достаточным числом затяжек. Сейчас **n = {int(n_target)}**, "
                f"**W = {Wk}** (D = {kin['D']}, Ш = {kin['Sh']}, C₀ = {kin['C0']}), затяжек: **{len(ties_all)}**."
            ],
            "latex_lines": [],
            "markdown_solution": [],
            "warnings": [],
            "kinematic_W": Wk,
            "tie_os_used": False,
            "tie_labels_ru": [],
        }

    if use_tie_os:
        n_req = n_need_tie
        removed = ties_all[:n_req]
        removed_ids = {t.id for t in removed}
        elements_os = [e for e in elements if e.id not in removed_ids]
        hinges_os = _hinges_excluding_elements(hinges, removed_ids)

        ss_os, w_os, _emap_os = build_system(
            nodes, elements_os, supports, [], hinges_os, global_ei=global_ei, global_ea=global_ea
        )
        warnings.extend(w_os)
        ok_os, err_os = try_solve(ss_os)
        if not ok_os:
            return {
                "ok": False,
                "markdown_intro": [f"**Основная система** (без затяжек) не решается: {err_os}"],
                "latex_lines": [],
                "markdown_solution": [],
                "warnings": warnings + [err_os],
                "kinematic_W": Wk,
                "tie_os_used": False,
                "tie_labels_ru": [],
            }

        ss_F, w_f, _ = build_system(
            nodes, elements_os, supports, loads, hinges_os, global_ei=global_ei, global_ea=global_ea
        )
        warnings.extend(w_f)
        ok_f, err_f = try_solve(ss_F)
        if not ok_f:
            return {
                "ok": False,
                "markdown_intro": [f"Грузовое состояние на основной системе (без затяжек) не решается: {err_f}"],
                "latex_lines": [],
                "markdown_solution": [],
                "warnings": warnings + [err_f],
                "kinematic_W": Wk,
                "tie_os_used": False,
                "tie_labels_ru": [],
            }

        unit_states: list[SystemElements] = []
        unit_labels_ru: list[str] = []
        tie_labels_ru: list[str] = []
        for t in removed:
            na, nb = _element_endpoint_names(t, nodes)
            seg = f"{na}—{nb}"
            tie_labels_ru.append(seg)
            try:
                uu = _solved_tie_cut_unit_system(ss_os, nodes, t)
            except Exception as exc:
                return {
                    "ok": False,
                    "markdown_intro": [
                        f"Не удалось построить единичное состояние для затяжки **{seg}**: `{exc}`."
                    ],
                    "latex_lines": [],
                    "markdown_solution": [],
                    "warnings": warnings + [str(exc)],
                    "kinematic_W": Wk,
                    "tie_os_used": False,
                    "tie_labels_ru": tie_labels_ru,
                }
            unit_states.append(uu)
            unit_labels_ru.append(
                f"разрез **затяжки** {na}—{nb}: встречные силы **1 кН** вдоль оси удалённого стержня"
            )

        n = n_req
        markdown_intro.append(
            f"По **Чебышёву**: **W = {Wk}** (D = {kin['D']}, Ш = {kin['Sh']}, C₀ = {kin['C0']}) → "
            f"**n ≈ {-Wk}**; вводим **{n}** неизвестных **X₁…X_{n}** как **усилия в снятых затяжках**."
        )
        markdown_intro.append(
            "**Основная система:** удалены стержни-затяжки (шарнир на обоих концах): "
            + ", ".join(f"**{lab}**" for lab in tie_labels_ru)
            + "."
        )
        markdown_intro.append(
            "Коэффициенты **δ_ij** — **∫(M̄_i M̄_j/EI)dx**; к **δ_ii** добавлена податливость снятой затяжки **L/(EA)** (N = 1)."
        )
        markdown_intro.append("**Соответствие неизвестных:**")
        for i, lab in enumerate(unit_labels_ru, start=1):
            markdown_intro.append(f"- **X_{i}** — {lab}")

        delta = np.zeros((n, n))
        DeltaF = np.zeros(n)
        for i in range(n):
            dfi, _rows, w = compute_mohr_displacement_report(ss_F, unit_states[i])
            DeltaF[i] = float(dfi)
            warnings.extend(w)
        for i in range(n):
            for j in range(n):
                val, _rows, w = compute_mohr_displacement_report(unit_states[i], unit_states[j])
                delta[i, j] = float(val)
                warnings.extend(w)

        delta = 0.5 * (delta + delta.T)
        for i in range(n):
            delta[i, i] += _tie_axial_self_flexibility(removed[i], nodes, global_ea)

        sd_map = support_settlement_rows_to_map(support_settlement_rows or [], supports)
        Delta_is, delta_is_detail = _force_method_delta_is_vector(unit_states, nodes, supports, sd_map, warnings)

        latex_lines.append(
            rf"\text{{Неизвестные (усилия в затяжках):}}\quad X_1,\,\ldots,\,X_{{{n}}}\quad \text{{(кН).}}"
        )
        _append_canonical_latex_with_settlements(
            latex_lines,
            n,
            delta,
            DeltaF,
            Delta_is,
            delta_is_detail,
            mohr_numeric_caption=r"\text{Численные значения (Мор + }L/(EA)\text{ на диагонали):}",
        )

        try:
            X = np.linalg.solve(delta, -(DeltaF + Delta_is))
        except np.linalg.LinAlgError:
            return {
                "ok": False,
                "markdown_intro": markdown_intro,
                "latex_lines": latex_lines,
                "markdown_solution": [
                    "**Матрица [δ] вырождена** — проверьте опоры, шарниры и выбор затяжек."
                ],
                "warnings": warnings,
                "kinematic_W": Wk,
                "tie_os_used": True,
                "tie_labels_ru": tie_labels_ru,
            }

        cond = float(np.linalg.cond(delta))
        if cond > 1.0e10:
            warnings.append(f"Матрица [δ] плохо обусловлена (cond ≈ {cond:.2e}) — решение может быть неточным.")

        markdown_solution.append("**Решение канонической системы** (численно, `numpy.linalg.solve`):")
        for i in range(n):
            markdown_solution.append(
                f"- **X_{i + 1}** = `{float(X[i]):.4f}` кН — усилие в разрезе затяжки **{tie_labels_ru[i]}**."
            )

        sup_terms = [rf"{_fmt_canonical_coef(float(X[j]))}\,\bar{{M}}_{{{j + 1}}}" for j in range(n)]
        latex_lines.append(r"\text{Суперпозиция изгибающих моментов (по стержням рамы):}")
        latex_lines.append(r"M_{\mathrm{ок}} \approx M_F + " + " + ".join(sup_terms))

        markdown_solution.append("")
        markdown_solution.append(
            "**Примечание:** суперпозиция по **M** — учебная оценка по стержням рамы; "
            "вклад **затяжек** учтён в **δ** и **Δ**."
        )

        uniq_warn = list(dict.fromkeys(warnings))
        x_list = [float(v) for v in X.tolist()]
        return {
            "ok": True,
            "markdown_intro": markdown_intro,
            "latex_lines": latex_lines,
            "markdown_solution": markdown_solution,
            "warnings": uniq_warn,
            "n_used": n,
            "n_target": int(-Wk),
            "X": x_list,
            "cond": cond,
            "kinematic_W": Wk,
            "tie_os_used": True,
            "tie_labels_ru": tie_labels_ru,
            "delta_is": [float(v) for v in Delta_is.tolist()],
            "delta_is_detail_latex": delta_is_detail,
        }

    if Wk < 0 and len(ties_all) < n_need_tie:
        warnings.append(
            f"По Чебышёву **W = {Wk}** (нужно **{-Wk}** избыточных связей), затяжек только **{len(ties_all)}** — "
            "используется подбор **единичных сил в узлах** (оценка **n** из редактора)."
        )

    name_to_id = node_name_to_model_node_id(ss_load_solved, nodes)
    available = sorted(name_to_id.keys())
    if not available:
        return {
            "ok": False,
            "markdown_intro": ["Нет узлов в расчётной модели — метод сил не применим."],
            "latex_lines": [],
            "markdown_solution": [],
            "warnings": warnings,
            "kinematic_W": Wk,
            "tie_os_used": False,
            "tie_labels_ru": [],
        }

    pairs: list[tuple[str, UnitDirection]] = []
    for nd in available:
        for d in ("y", "x", "rz"):
            pairs.append((nd, d))

    n_req = min(int(n_target), int(max_unknowns), len(pairs))
    if n_req < int(n_target):
        warnings.append(
            f"По редактору **n = {n_target}**, но автоматически сформировано только **{n_req}** "
            f"единичных состояний (ограничение {max_unknowns} или число комбинаций «узел × направление»). "
            f"Матрица гибкости имеет размер **{n_req}×{n_req}**."
        )

    dir_ru = {
        "y": "вертикальная единичная сила 1 кН (как в расчёте перемещений)",
        "x": "горизонтальная единичная сила 1 кН",
        "rz": "единичный момент 1 кН·м",
    }
    unit_states: list[SystemElements] = []
    unit_labels_ru: list[str] = []
    for k in range(n_req):
        node, d = pairs[k]
        try:
            ssu = _solved_unit_system_for_force_method(ss_load_solved, nodes, node, d)
        except Exception as exc:
            return {
                "ok": False,
                "markdown_intro": [f"Не удалось построить единичное состояние **{k + 1}**: `{exc}`."],
                "latex_lines": [],
                "markdown_solution": [],
                "warnings": warnings + [str(exc)],
                "kinematic_W": Wk,
                "tie_os_used": False,
                "tie_labels_ru": [],
            }
        unit_states.append(ssu)
        unit_labels_ru.append(f"узел **{node}**, {dir_ru[d]}")

    n = n_req
    markdown_intro.append(
        f"Степень статической неопределимости (по оценке редактора): **n = {n_target}**. "
        f"Вводим **{n}** избыточных неизвестных **X₁ … X_{n}** (силы или моменты в «лишних» связях в учебной постановке)."
    )
    markdown_intro.append(
        f"Кинематика (Чебышёв): **W = {Wk}** (D = {kin['D']}, Ш = {kin['Sh']}, C₀ = {kin['C0']})."
    )
    markdown_intro.append(
        "Коэффициенты **δ_ij** (взаимные перемещения по направлениям избыточных) и свободные члены "
        "**Δ_iF** (перемещения от внешней нагрузки) вычислены **численно** как интегралы Мора "
        "∫(M·M'/EI)dx по стержням (как при расчёте перемещений)."
    )
    markdown_intro.append("**Соответствие неизвестных единичным воздействиям:**")
    for i, lab in enumerate(unit_labels_ru, start=1):
        markdown_intro.append(f"- **X_{i}** — {lab}")

    delta = np.zeros((n, n))
    DeltaF = np.zeros(n)
    for i in range(n):
        dfi, _rows, w = compute_mohr_displacement_report(ss_load_solved, unit_states[i])
        DeltaF[i] = float(dfi)
        warnings.extend(w)
    for i in range(n):
        for j in range(n):
            val, _rows, w = compute_mohr_displacement_report(unit_states[i], unit_states[j])
            delta[i, j] = float(val)
            warnings.extend(w)

    delta = 0.5 * (delta + delta.T)

    sd_map = support_settlement_rows_to_map(support_settlement_rows or [], supports)
    Delta_is, delta_is_detail = _force_method_delta_is_vector(unit_states, nodes, supports, sd_map, warnings)

    latex_lines.append(
        rf"\text{{Неизвестные:}}\quad X_1,\,X_2,\,\ldots,\,X_{{{n}}}\quad"
        rf"\text{{ (силы в кН или моменты в кН·м в зависимости от единицы).}}"
    )
    _append_canonical_latex_with_settlements(
        latex_lines,
        n,
        delta,
        DeltaF,
        Delta_is,
        delta_is_detail,
        mohr_numeric_caption=r"\text{Численные значения коэффициентов (интеграл Мора):}",
    )

    try:
        X = np.linalg.solve(delta, -(DeltaF + Delta_is))
    except np.linalg.LinAlgError:
        return {
            "ok": False,
            "markdown_intro": markdown_intro,
            "latex_lines": latex_lines,
            "markdown_solution": [
                "**Матрица коэффициентов [δ] вырождена** — выбранные автоматически единичные состояния, "
                "скорее всего, линейно зависимы для данной схемы. Попробуйте изменить опоры, шарниры или схему."
            ],
            "warnings": warnings,
            "kinematic_W": Wk,
            "tie_os_used": False,
            "tie_labels_ru": [],
        }

    cond = float(np.linalg.cond(delta))
    if cond > 1.0e10:
        warnings.append(f"Матрица [δ] плохо обусловлена (cond ≈ {cond:.2e}) — решение может быть неточным.")

    markdown_solution.append("**Решение канонической системы** (численно, `numpy.linalg.solve`):")
    for i in range(n):
        node_i, dcode = pairs[i]
        unit_txt = "кН·м" if dcode == "rz" else "кН"
        markdown_solution.append(
            f"- **X_{i + 1}** = `{float(X[i]):.4f}` {unit_txt} — неизвестная, отвечающая единице в узле **{node_i}** "
            f"({'момент' if dcode == 'rz' else 'сила'})."
        )

    sup_terms = [rf"{_fmt_canonical_coef(float(X[j]))}\,\bar{{M}}_{{{j + 1}}}" for j in range(n)]
    latex_lines.append(r"\text{Суперпозиция изгибающих моментов:}")
    latex_lines.append(r"M_{\mathrm{ок}} = M_F + " + " + ".join(sup_terms))

    markdown_solution.append("")
    markdown_solution.append(
        "**Принцип суперпозиции:** итоговая эпюра моментов в линейно упругой постановке записывается как "
        "**M_ок = M_F + X₁·M̄₁ + … + Xₙ·M̄ₙ**, где **M_F** — моменты от заданной нагрузки, **M̄ⱼ** — от j-го единичного состояния "
        "(без учёта масштаба; множитель **Xⱼ** подобран из канонической системы)."
    )

    uniq_warn = list(dict.fromkeys(warnings))
    x_list: list[float] = [float(v) for v in X.tolist()]
    return {
        "ok": True,
        "markdown_intro": markdown_intro,
        "latex_lines": latex_lines,
        "markdown_solution": markdown_solution,
        "warnings": uniq_warn,
        "n_used": n,
        "n_target": int(n_target),
        "X": x_list,
        "cond": cond,
        "kinematic_W": Wk,
        "tie_os_used": False,
        "tie_labels_ru": [],
        "delta_is": [float(v) for v in Delta_is.tolist()],
        "delta_is_detail_latex": delta_is_detail,
    }
