"""Streamlit UI: sidebar, tabs, main flow."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from anastruct import SystemElements

import engine
import plotting
import state_manager as sm
from models import Node, MemberElement, SupportEntry, HingeEntry

_DISPLACEMENT_TAB_MODES: tuple[str, ...] = (
    "Только силовое (Метод Мора)",
    "Осадки опор",
    "Температурное (позже)",
)


def _support_displacements_zeroed() -> dict[int, dict]:
    """Копия осадок с обнулёнными перемещениями (режим «только Мор»)."""
    sm.ensure_support_displacements_state()
    out: dict[int, dict] = {}
    for support in st.session_state.supports:
        sid = int(support["id"])
        node = str(support["node"])
        stype = support["type"]
        cur = st.session_state.support_displacements.get(sid, {})
        if stype == "roller":
            out[sid] = {
                "sym_dn": str(cur.get("sym_dn", f"c_{node}n")).strip() or f"c_{node}n",
                "dn_mm": 0.0,
                "dx_mm": 0.0,
                "dy_mm": 0.0,
                "phi_rad": 0.0,
            }
        elif stype == "hinged":
            out[sid] = {
                "sym_dx": str(cur.get("sym_dx", f"c_{node}x")).strip() or f"c_{node}x",
                "sym_dy": str(cur.get("sym_dy", f"c_{node}y")).strip() or f"c_{node}y",
                "dx_mm": 0.0,
                "dy_mm": 0.0,
                "phi_rad": 0.0,
            }
        else:
            out[sid] = {
                "sym_dx": str(cur.get("sym_dx", f"c_{node}x")).strip() or f"c_{node}x",
                "sym_dy": str(cur.get("sym_dy", f"c_{node}y")).strip() or f"c_{node}y",
                "sym_phi": str(cur.get("sym_phi", f"φ_{node}")).strip() or f"φ_{node}",
                "dx_mm": 0.0,
                "dy_mm": 0.0,
                "phi_rad": 0.0,
            }
    return out


def _editor_model_bundle() -> tuple:
    """Current editor geometry and loads from Streamlit session (passed into pure engine)."""
    return (
        st.session_state.nodes,
        st.session_state.elements,
        st.session_state.supports,
        st.session_state.loads,
        st.session_state.hinges,
        float(st.session_state.get("global_EI", 5000.0)),
        float(st.session_state.get("global_EA", 1.0e9)),
    )


def local_css(file_name: str) -> None:
    """Reads local CSS file and injects it into the app."""
    path = Path(file_name)
    if path.is_file():
        st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_loads_summary() -> None:
    """Compact summary of user-defined loads."""
    if not st.session_state.loads:
        st.info("Заданные нагрузки: пока нет.")
        return

    lines: list[str] = []
    for load in st.session_state.loads:
        lines.append(f"- {sm.format_load_short(load)}")

    st.markdown("#### Заданные нагрузки")
    st.info("\n".join(lines))

def _russian_degrees_of_freedom_phrase(w: int) -> str:
    """Склонение для фразы «N степен(ь/и/ей) свободы»."""
    w = abs(int(w))
    n10 = w % 10
    n100 = w % 100
    if n10 == 1 and n100 != 11:
        return f"{w} степень свободы"
    if 2 <= n10 <= 4 and (n100 < 10 or n100 >= 20):
        return f"{w} степени свободы"
    return f"{w} степеней свободы"


def render_kinematic_analysis(
    n_val: int,
    sop: int,
    support_counts: dict[str, int],
    k_loops: int,
) -> None:
    from typing import cast
    from models import Node, MemberElement, SupportEntry, HingeEntry

    nodes = cast(list[Node], st.session_state.nodes)
    elements = cast(list[MemberElement], st.session_state.elements)
    supports = cast(list[SupportEntry], st.session_state.supports)
    hinges = cast(list[HingeEntry], st.session_state.hinges)

    ka = engine.calculate_kinematic_analysis(nodes, elements, supports, hinges)
    W, D, Sh, C0 = ka["W"], ka["D"], ka["Sh"], ka["C0"]

    if W == 0:
        verdict = "Система может быть статически определимой и геометрически неизменяемой"
    elif W > 0:
        verdict = f"Механизм, {W} степен(ь/и/ей) свободы"
    else:
        verdict = f"Система статически неопределима, n = {-W}"

    formula = f"W = 3·D - 2·Ш - C₀ = 3·{D} - 2·{Sh} - {C0} = {W}"
    st.markdown(f"{formula} → **{verdict}**")


def render_static_determinacy_block() -> None:
    """Compact status line for static determinacy."""
    n, el, sup, ld, hn, _ei, _ea = _editor_model_bundle()
    n_val, sop, cnt, k_loops = engine.calculate_static_indeterminacy_n(n, el, sup, hn)
    ka = engine.calculate_kinematic_analysis(n, el, sup, hn)
    Wk = int(ka["W"])
    if Wk < 0:
        st.caption(f"Кинематика (Чебышёв): система статически неопределима, **W = {Wk}**.")
    elif Wk > 0:
        st.caption(f"Кинематика (Чебышёв): механизм, **W = {Wk}**.")
    else:
        st.caption(f"Кинематика (Чебышёв): **W = 0** (D = {ka['D']}, Ш = {ka['Sh']}, C₀ = {ka['C0']}).")
    if st.session_state.elements:
        st.caption(
            f"Замкнутые контуры по графу стержней: K = {k_loops}; "
            f"опорных связей Sop = {sop} "
            f"(жёсткая {cnt['fixed']}, шарнирная {cnt['hinged']}, катящаяся {cnt['roller']})."
        )
    if n_val == 0:
        st.success("✅ Система статически определима (n = 0)")
    elif n_val > 0:
        st.warning(f"❗ Система статически неопределима (n = {n_val})")
    else:
        render_kinematic_analysis(n_val, sop, cnt, k_loops)


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


def _support_type_ru(code: str) -> str:
    return {"fixed": "Жёсткая", "hinged": "Шарнирная", "roller": "Качельная"}.get(str(code), str(code))


def render_support_settlements_table_and_force_method(solved_ss: SystemElements | None) -> None:
    """Таблица осадок опор + применение в состояние + отчёт метода сил с учётом Δ_is."""
    st.markdown("##### Осадки опор")
    if not st.session_state.supports:
        st.info("Опоры не заданы.")
        return

    sm.ensure_support_displacements_state()
    rows = list(st.session_state.support_settlement_rows or [])
    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("Не удалось сформировать строки осадок.")
        return
    df["support_type_ru"] = df["support_type"].map(_support_type_ru)

    edited = st.data_editor(
        df,
        column_config={
            "support_id": st.column_config.NumberColumn("ID опоры", disabled=True, format="%d"),
            "support_type": st.column_config.TextColumn("Код типа", disabled=True),
            "node_name": st.column_config.TextColumn("Узел", disabled=True),
            "support_type_ru": st.column_config.TextColumn("Тип опоры", disabled=True),
            "direction": st.column_config.TextColumn("Направление", disabled=True),
            "symbol": st.column_config.TextColumn("Обозначение (для формулы)"),
            "value": st.column_config.NumberColumn("Величина (м или рад)", format="%.6f", min_value=None),
        },
        column_order=["node_name", "support_type_ru", "direction", "symbol", "value"],
        hide_index=True,
        num_rows="fixed",
        key="support_settlements_data_editor",
    )

    # Сразу переносим правки таблицы в session_state — иначе расчёт Δ_c / метод сил видят старые нули,
    # пока не нажата отдельная кнопка (st.data_editor сам по себе не пишет в support_settlement_rows).
    sm.commit_support_settlement_from_dataframe(edited.drop(columns=["support_type_ru"], errors="ignore"))

    st.caption(
        "Направление **X**, **Y** — линейные смещения в метрах; **Rot** — поворот жёсткой заделки в радианах; "
        "**N** — смежение вдоль нормали качельной связи (м). Обозначения используются в LaTeX отчёта. "
        "Введённые значения **сразу** участвуют в расчётах ниже (Δ_c, метод сил)."
    )

    if solved_ss is None:
        st.info("Решите основную систему на вкладке «Расчёт», чтобы построить отчёт метода сил с учётом осадок.")
        return

    st.markdown("---")
    st.markdown("##### Метод сил (с учётом осадок опор, Δ_is)")
    _n_fm, _el_fm, sup_fm, ld_fm, hn_fm, ei_fm, ea_fm = _editor_model_bundle()
    fm = engine.build_force_method_report(
        solved_ss,
        st.session_state.nodes,
        _el_fm,
        sup_fm,
        hn_fm,
        ld_fm,
        max_unknowns=10,
        global_ei=ei_fm,
        global_ea=ea_fm,
        support_settlement_rows=list(st.session_state.get("support_settlement_rows") or []),
    )
    for w in fm.get("warnings", []):
        st.warning(w)
    if fm.get("tie_os_used"):
        st.success(
            "Метод сил: **основная система без затяжек**; неизвестные **Xᵢ** — усилия в разрезах: "
            + ", ".join(f"**{lab}**" for lab in fm.get("tie_labels_ru", []))
            + "."
        )
    for line in fm.get("markdown_intro", []):
        st.markdown(line)
    if fm.get("latex_lines"):
        st.markdown("**Канонические уравнения и суперпозиция (LaTeX):**")
        for lx in fm["latex_lines"]:
            st.latex(lx)
    for line in fm.get("markdown_solution", []):
        if line.strip() == "":
            st.markdown("")
        else:
            st.markdown(line)


def render_temperature_expansion_inputs() -> None:
    """Symbolic label + value for thermal expansion coefficient (displacements tab only)."""
    st.markdown("##### Температурный коэффициент")
    if "displacement_tab_params" not in st.session_state:
        st.session_state.displacement_tab_params = {"thermal_tau": 0.0, "thermal_tau_label": "τ"}
    p = st.session_state.displacement_tab_params
    c1, c2 = st.columns(2)
    with c1:
        tau_label = st.text_input(
            "Обозначение (например τ или α_t)",
            value=str(p.get("thermal_tau_label", "τ")),
            key="disp_thermal_tau_label",
        )
    with c2:
        tau_val = st.number_input(
            "Значение, 1/°C",
            value=float(p.get("thermal_tau", 0.0)),
            format="%.2e",
            step=1e-7,
            key="disp_thermal_tau_value",
        )
    st.session_state.displacement_tab_params = {
        "thermal_tau_label": str(tau_label).strip() or "τ",
        "thermal_tau": float(tau_val),
    }
    st.caption(
        "Параметр хранится в session_state для учебного отчёта; в численный интеграл Мора "
        "текущая версия приложения его не включает."
    )


def result_card(ss: SystemElements, reaction_rows: list[dict]) -> None:
    max_m = float(max(ss.get_element_result_range("moment", "abs") or [0.0]))
    max_q = float(max(ss.get_element_result_range("shear", "abs") or [0.0]))
    max_n = float(max(ss.get_element_result_range("axial", "abs") or [0.0]))
    st.markdown(
        f"""
        <div class="card bg-dark text-white border-warning forge-shadow mb-3" style="border-width:2px;">
            <div class="card-header border-warning text-warning fw-bold">Ключевые результаты расчёта</div>
            <div class="card-body py-3">
                <div class="row g-3">
                    <div class="col-md-4"><div class="text-secondary small">Max |M|</div><div class="fs-4">{max_m:.3f} кН·м</div></div>
                    <div class="col-md-4"><div class="text-secondary small">Max |Q|</div><div class="fs-4">{max_q:.3f} кН</div></div>
                    <div class="col-md-4"><div class="text-secondary small">Max |N|</div><div class="fs-4">{max_n:.3f} кН</div></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if reaction_rows:
        lines = []
        for row in reaction_rows:
            m_line = ""
            if row["Тип опоры (код)"] == "fixed" or abs(float(row["Mz (Tz)"])) > 1e-8:
                m_line = f", **M{row['Индекс']}**={row['Mz (Tz)']:.3f} кН·м"
            lines.append(
                f"- **{row['Узел']}**: **R{row['Индекс']}x**={row['Fx']:.3f} кН ({row['Направление Fx']}), "
                f"**R{row['Индекс']}y**={row['Fy']:.3f} кН ({row['Направление Fy']}){m_line}"
            )
        st.markdown("#### Реакции опор (сводно)")
        st.markdown("\n".join(lines))

def render_support_reactions(ss: SystemElements) -> None:
    """Render support reactions table and reaction-force plot."""
    st.markdown("#### Реакции опор")
    if not st.session_state.supports:
        st.info("Опоры не заданы. Реакции отсутствуют.")
        return

    reaction_rows, missed = engine.collect_support_reaction_rows(
        ss, st.session_state.nodes, st.session_state.supports
    )

    if reaction_rows:
        st.dataframe(reaction_rows, use_container_width=True, hide_index=True)
        symbolic_lines = []
        for row in reaction_rows:
            symbolic_lines.append(f"- Узел **{row['Узел']}**: `R{row['Индекс']}x = {row['Fx']:.3f} кН`")
            symbolic_lines.append(f"  `R{row['Индекс']}y = {row['Fy']:.3f} кН`")
            if row["Тип опоры (код)"] == "fixed" or abs(float(row["Mz (Tz)"])) > 1e-8:
                symbolic_lines.append(f"  `M{row['Индекс']} = {row['Mz (Tz)']:.3f} кН·м`")
        st.markdown("#### Реакции с буквенными индексами")
        st.markdown("\n".join(symbolic_lines))
    else:
        st.warning("Не удалось сформировать таблицу реакций для текущей модели.")

    if missed:
        st.warning("Часть опор не вошла в таблицу:\n- " + "\n- ".join(missed))

    try:
        fig_react = ss.show_reaction_force(show=False, verbosity=1, offset=(0.0, 0.05))
        plotting.enhance_anastruct_figure(fig_react, arrow_color="#D32F2F", arrow_linewidth=2.6)
        st.pyplot(fig_react, clear_figure=True)
    except Exception as exc:
        st.error(f"Не удалось построить график реакций: {exc}")


def render_sidebar() -> bool:
    with st.sidebar:
        st.markdown("## Редактор модели")

        # Nodes
        with st.expander("1) Узлы", expanded=True):
            with st.form("add_node_form", clear_on_submit=True):
                default_name = sm.get_next_node_label()
                wkey = int(st.session_state.get("node_name_widget_key", 0))
                node_name = st.text_input(
                    "Имя узла",
                    value=default_name,
                    placeholder="A",
                    key=f"add_node_name_{wkey}",
                    help="По умолчанию — следующая буква в серии A…Z, A1… Можно стереть и ввести своё имя.",
                )
                col1, col2 = st.columns(2)
                with col1:
                    x = st.number_input("X", value=0.0, step=0.5, format="%.2f", key="node_x")
                with col2:
                    y = st.number_input("Y", value=0.0, step=0.5, format="%.2f", key="node_y")
                if st.form_submit_button("Добавить узел"):
                    ok, msg = sm.add_node(node_name, x, y)
                    if ok:
                        st.session_state.node_name_widget_key = wkey + 1
                        st.toast(msg, icon="✅")
                        st.rerun()
                    else:
                        st.error(msg)

            if st.session_state.nodes:
                node_options = [n.name for n in st.session_state.nodes]
                node_to_delete = st.selectbox("Удалить узел", options=node_options, key="node_to_delete")
                if st.button("Удалить выбранный узел"):
                    ok, msg = sm.delete_node(node_to_delete)
                    (st.success if ok else st.error)(msg)
                st.dataframe(sm.nodes_as_table_rows(), use_container_width=True, hide_index=True)
            else:
                st.info("Узлы пока не добавлены.")

        # Elements
        with st.expander("2) Стержни", expanded=True):
            node_names = [n.name for n in st.session_state.nodes]
            if len(node_names) >= 2:
                with st.form("add_element_form", clear_on_submit=True):
                    start = st.selectbox("Начальный узел", options=node_names, key="elem_start")
                    end = st.selectbox("Конечный узел", options=node_names, key="elem_end")
                    if st.form_submit_button("Добавить стержень"):
                        ok, msg = sm.add_element(start, end)
                        (st.success if ok else st.error)(msg)
            else:
                st.info("Добавьте минимум 2 узла.")

            if st.session_state.elements:
                element_ids = [e.id for e in st.session_state.elements]
                by_id = {e.id: e for e in st.session_state.elements}
                label = st.selectbox(
                    "Удалить стержень",
                    options=element_ids,
                    format_func=lambda eid: sm._element_ui_label(by_id[eid]),
                    key="element_to_delete",
                )
                if st.button("Удалить выбранный стержень"):
                    ok, msg = sm.delete_element(label)
                    (st.success if ok else st.error)(msg)
                st.dataframe(sm.elements_as_table_rows(), use_container_width=True, hide_index=True)

                st.markdown("##### Параметры стержня")
                tie_target = st.selectbox(
                    "Стержень",
                    options=element_ids,
                    format_func=lambda eid: sm._element_ui_label(by_id[eid]),
                    key="element_tie_target",
                )
                tie_value = st.checkbox(
                    "Затяжка",
                    value=bool(by_id[tie_target].is_tie),
                    key=f"element_is_tie_{str(tie_target)}",
                    help="Для затяжки автоматически добавляются шарниры на начале и конце стержня.",
                )
                if st.button("Применить параметр стержня", key="apply_element_tie"):
                    ok, msg = sm.set_element_tie(tie_target, tie_value)
                    (st.success if ok else st.error)(msg)
            else:
                st.info("Стержни пока не добавлены.")

        # Supports
        with st.expander("3) Опоры", expanded=True):
            node_names = [n.name for n in st.session_state.nodes]
            if node_names:
                with st.form("add_support_form", clear_on_submit=True):
                    node = st.selectbox("Узел", options=node_names, key="support_node")
                    support_type = st.selectbox(
                        "Тип опоры",
                        options=[
                            ("fixed", "Жесткая заделка (Fixed)"),
                            ("hinged", "Шарнирно-неподвижная (Hinged)"),
                            ("roller", "Шарнирно-подвижная (Roller)"),
                        ],
                        format_func=lambda x: x[1],
                        key="support_type",
                    )[0]
                    if support_type == "fixed":
                        st.caption(
                            "Для **заделки** угол наклона **не задаётся** в anaStruct и **не меняет** расчётную схему."
                        )
                    angle = st.number_input(
                        "Угол наклона связи, ° (от оси X против часовой)",
                        value=0.0,
                        step=5.0,
                        key="support_angle",
                        disabled=support_type == "fixed",
                        help=(
                            "**Roller:** угол направления катящейся связи (передаётся в anaStruct). "
                            "**Hinged:** в anaStruct у шарнирно-неподвижной опоры угла нет — значение "
                            "сохраняется для отображения; на схему МКЭ не влияет."
                        ),
                    )
                    if st.form_submit_button("Добавить/заменить опору"):
                        ok, msg = sm.add_support(node, support_type, float(angle))
                        if ok:
                            st.toast(msg, icon="✅")
                            st.rerun()
                        else:
                            st.error(msg)
            else:
                st.info("Сначала добавьте узлы.")

            if st.session_state.supports:
                support_labels = [
                    f"S{s['id']}: {s['node']} [{s['type']}"
                    + (
                        f", angle={s.get('angle', 0)}°"
                        if s["type"] in ("roller", "hinged")
                        else ""
                    )
                    + "]"
                    for s in st.session_state.supports
                ]
                label = st.selectbox("Удалить опору", options=support_labels, key="support_to_delete")
                if st.button("Удалить выбранную опору"):
                    support_id = int(label.split(":")[0][1:])
                    ok, msg = sm.delete_support(support_id)
                    (st.success if ok else st.error)(msg)
                st.dataframe(st.session_state.supports, use_container_width=True, hide_index=True)
            else:
                st.info("Опоры пока не добавлены.")

        # Loads
        with st.expander("4) Нагрузки", expanded=True):
            load_kind = st.radio(
                "Тип нагрузки",
                options=["point", "distributed", "moment"],
                horizontal=True,
                format_func=lambda x: {
                    "point": "Сосредоточенная сила",
                    "distributed": "Распределенная",
                    "moment": "Сосредоточенный момент",
                }[x],
            )

            node_names = [n.name for n in st.session_state.nodes]
            if load_kind == "point":
                if node_names:
                    with st.form("add_point_load_form", clear_on_submit=True):
                        node = st.selectbox("Узел", options=node_names, key="point_node")
                        fx = st.number_input("Fx", value=0.0, step=1.0, key="point_fx")
                        fy = st.number_input("Fy", value=0.0, step=1.0, key="point_fy")
                        if st.form_submit_button("Добавить нагрузку"):
                            ok, msg = sm.add_load({"type": "point", "node": node, "Fx": float(fx), "Fy": float(fy)})
                            (st.success if ok else st.error)(msg)
                else:
                    st.info("Нет узлов для задания нагрузки.")

            elif load_kind == "distributed":
                if st.session_state.elements:
                    with st.form("add_distributed_load_form", clear_on_submit=True):
                        element_ids = [e.id for e in st.session_state.elements]
                        by_id = {e.id: e for e in st.session_state.elements}
                        picked_id = st.selectbox(
                            "Стержень",
                            options=element_ids,
                            format_func=lambda eid: sm._element_ui_label(by_id[eid]),
                            key="q_element",
                        )
                        q = st.number_input("q (отрицательное = вниз)", value=-5.0, step=1.0, key="q_value")
                        if st.form_submit_button("Добавить нагрузку"):
                            ok, msg = sm.add_load({"type": "distributed", "element_id": picked_id, "q": float(q)})
                            (st.success if ok else st.error)(msg)
                else:
                    st.info("Нет стержней для задания распределенной нагрузки.")

            elif load_kind == "moment":
                if node_names:
                    with st.form("add_moment_load_form", clear_on_submit=True):
                        node = st.selectbox("Узел", options=node_names, key="moment_node")
                        mz = st.number_input("Момент Mz", value=5.0, step=1.0, key="moment_value")
                        if st.form_submit_button("Добавить нагрузку"):
                            ok, msg = sm.add_load({"type": "moment", "node": node, "M": float(mz)})
                            (st.success if ok else st.error)(msg)
                else:
                    st.info("Нет узлов для задания момента.")

            if st.session_state.loads:
                load_ids = [int(load["id"]) for load in st.session_state.loads]
                labels_by_id = {int(load["id"]): sm.format_load_short(load) for load in st.session_state.loads}
                selected_load_id = st.selectbox(
                    "Удалить нагрузку",
                    options=load_ids,
                    format_func=lambda lid: labels_by_id.get(lid, f"Нагрузка {lid}"),
                    key="load_to_delete",
                )
                if st.button("Удалить выбранную нагрузку"):
                    ok, msg = sm.delete_load(selected_load_id)
                    (st.success if ok else st.error)(msg)
                st.dataframe(st.session_state.loads, use_container_width=True, hide_index=True)
            else:
                st.info("Нагрузки пока не добавлены.")

        # Hinges
        with st.expander("5) Шарниры", expanded=False):
            if st.session_state.elements:
                node_names = [n.name for n in st.session_state.nodes]
                if node_names:
                    with st.form("add_hinge_form", clear_on_submit=True):
                        hinge_node = st.selectbox("Узел", options=node_names, key="hinge_node_name")
                        if st.form_submit_button("Добавить шарниры в узле"):
                            ok, msg = sm.add_hinges_at_node(hinge_node)
                            if ok:
                                st.success(msg)
                            elif msg == "Узел не принадлежит ни одному стержню":
                                st.warning(msg)
                            else:
                                st.error(msg)
                else:
                    st.info("Сначала добавьте узлы.")
            else:
                st.info("Сначала добавьте стержни.")

            if st.session_state.hinges:
                st.markdown("**Шарниры по узлам**")
                st.caption("Удаление только целиком по узлу: снимаются все шарниры, относящиеся к выбранному узлу.")

                by_node: dict[str, list[dict]] = defaultdict(list)
                orphans: list[dict] = []
                for h in list(st.session_state.hinges):
                    nn = sm._hinge_node_name(h)
                    if nn is None:
                        orphans.append(h)
                    else:
                        by_node[nn].append(h)
                for node_nm in sorted(by_node.keys()):
                    hinges_here = by_node[node_nm]
                    row_l, row_r = st.columns([3, 2], gap="small")
                    with row_l:
                        detail = " · ".join(sm._hinge_ui_label(x) for x in hinges_here)
                        st.markdown(f"**Узел {node_nm}** ({len(hinges_here)} шт.)")
                        st.caption(detail)
                    with row_r:
                        if st.button(
                            "Удалить все в узле",
                            key=f"delete_hinges_node_{node_nm}",
                            use_container_width=True,
                        ):
                            ok, msg = sm.delete_hinges_at_node(node_nm)
                            (st.success if ok else st.error)(msg)
                if orphans:
                    row_lo, row_ro = st.columns([3, 2], gap="small")
                    with row_lo:
                        st.markdown("**Без узла**")
                        st.caption(
                            "Шарниры не сопоставлены с узлом (например, стержень удалён). "
                            + " · ".join(f"id {h.get('id')}" for h in orphans)
                        )
                    with row_ro:
                        if st.button(
                            "Удалить без узла",
                            key="delete_hinges_orphan",
                            use_container_width=True,
                        ):
                            ok, msg = sm.delete_orphan_hinges()
                            (st.success if ok else st.error)(msg)
            else:
                st.info("Шарниры не добавлены.")

        with st.expander("6) Свойства материалов", expanded=False):
            st.caption("Жёсткости задаются в кН·м² (EI) и кН (EA) и применяются ко всем стержням при сборке модели.")
            st.number_input(
                "EI (изгибная жесткость, кН·м²)",
                min_value=1e-12,
                value=float(st.session_state.global_EI),
                step=100.0,
                format="%.6g",
                key="global_EI",
            )
            st.number_input(
                "EA (продольная жесткость, кН)",
                min_value=1e-12,
                value=float(st.session_state.global_EA),
                step=1.0e6,
                format="%.6g",
                key="global_EA",
            )

        st.session_state.setdefault("epure_scale", 0.65)
        st.slider(
            "Масштаб эпюр (M, Q, N)",
            min_value=0.15,
            max_value=1.5,
            step=0.05,
            key="epure_scale",
            help="Параметр **scale** в anaStruct: больше — крупнее эпюры относительно схемы.",
        )

        st.markdown("### Сводка по нагрузкам")
        render_loads_summary()
        st.markdown("---")
        return st.button("Узнать горькую правду", type="primary", use_container_width=True)
def main() -> None:
    st.set_page_config(page_title="Plane Frame Solver", page_icon="📐", layout="wide", initial_sidebar_state="expanded")
    plotting.configure_matplotlib_style()
    st.markdown(
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">',
        unsafe_allow_html=True,
    )
    local_css("style.css")
    sm.init_state()

    st.markdown('<div class="app-title">Редактор плоских рам (Streamlit + anaStruct)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Твои дома обязательно будут стоять.</div>',
        unsafe_allow_html=True,
    )

    calculate_pressed = render_sidebar()

    # Live preview (без solve): узлы, стержни, опоры и нагрузки — всё из session_state
    n, el, sup, ld, hn, ei, ea = _editor_model_bundle()
    preview_ss, preview_warnings, _ = engine.build_system(n, el, sup, ld, hn, global_ei=ei, global_ea=ea)
    st.markdown("### Предварительная схема (live)")
    render_static_determinacy_block()
    plotting.render_live_preview(preview_ss)

    if preview_warnings:
        st.warning("Некоторые объекты не отображены в превью:\n- " + "\n- ".join(preview_warnings))

    if calculate_pressed:
        if not st.session_state.elements:
            st.error("Расчет невозможен: нет стержней.")
        elif not st.session_state.supports:
            st.error("Расчет невозможен: не задано ни одной опоры.")
        else:
            solved_ss, solve_warnings, el_map = engine.build_system(n, el, sup, ld, hn, global_ei=ei, global_ea=ea)
            ok_solve, solve_err = engine.try_solve(solved_ss)
            if not ok_solve:
                try:
                    validation_data = solved_ss.validate()
                    if validation_data:
                        st.warning("Диагностика системы:")
                        st.write(validation_data)
                    else:
                        st.info("validate() не вернул замечаний.")
                except Exception as exc:
                    st.warning(f"Не удалось выполнить validate(): {exc}")


                st.error(solve_err)
                st.session_state.solved_ss = None
                st.session_state.solve_warnings = solve_warnings
                st.session_state.session_to_ana_map = None
                st.session_state.unit_result = None
            else:
                st.session_state.solved_ss = solved_ss
                st.session_state.solve_warnings = solve_warnings
                st.session_state.session_to_ana_map = el_map
                st.session_state.unit_result = None

    solved_ss = st.session_state.solved_ss
    if solved_ss is None:
        return

    st.markdown("### Результаты расчета")
    if st.session_state.solve_warnings:
        st.warning("Некоторые объекты были пропущены при расчете:\n- " + "\n- ".join(st.session_state.solve_warnings))

    tab_eff, tab_disp, tab_force = st.tabs(["Усилия и реакции", "Перемещения", "Метод сил"])

    with tab_eff:
        reaction_rows, reaction_missed = engine.collect_support_reaction_rows(
            solved_ss, st.session_state.nodes, st.session_state.supports
        )
        render_static_determinacy_block()
        result_card(solved_ss, reaction_rows)
        if reaction_missed:
            st.warning("Часть реакций не удалось сопоставить:\n- " + "\n- ".join(reaction_missed))

        st.markdown("#### Эпюры M, Q, N")
        st.caption(
            "Числовые подписи: в anaStruct для этого используется **verbosity=0**; масштаб эпюр — слайдер в боковой панели."
        )
        _ep = plotting.epure_diagram_kwargs(float(st.session_state.get("epure_scale", 0.65)), values_deflection=1.0)
        c1, c2, c3 = st.columns(3)
        with c1:
            fig_m = solved_ss.show_bending_moment(**_ep)
            plotting.enhance_anastruct_figure(
                fig_m, arrow_color="#D32F2F", arrow_linewidth=2.4, annotation_fontsize=10
            )
            st.pyplot(fig_m, clear_figure=True)
        with c2:
            fig_q = solved_ss.show_shear_force(**_ep)
            plotting.enhance_anastruct_figure(
                fig_q, arrow_color="#D32F2F", arrow_linewidth=2.4, annotation_fontsize=10
            )
            st.pyplot(fig_q, clear_figure=True)
        with c3:
            fig_n = solved_ss.show_axial_force(**_ep)
            plotting.enhance_anastruct_figure(
                fig_n, arrow_color="#D32F2F", arrow_linewidth=2.4, annotation_fontsize=10
            )
            st.pyplot(fig_n, clear_figure=True)

        with st.expander("Показать пошаговый ход решения (Метод сечений)", expanded=False):
            step_out = engine.generate_step_by_step_analysis(
                st.session_state.nodes,
                st.session_state.elements,
                st.session_state.loads,
                st.session_state.supports,
                solved_ss,
                hinges=st.session_state.hinges,
            )
            if isinstance(step_out, str):
                st.info(step_out)
            else:
                for blk in step_out:
                    for line in blk.get("markdown", []):
                        st.markdown(line, unsafe_allow_html=False)
                    for line in blk.get("latex", []):
                        st.latex(line)

        render_support_reactions(solved_ss)

    with tab_disp:
        st.markdown("#### Перемещения")
        st.caption(
            "Выберите режим: **силовое** (только метод Мора), **осадки** (Δ = Δ_P + Δ_c) "
            "или **температурное** (в разработке)."
        )
        mode = st.radio(
            "Режим расчёта перемещений:",
            options=list(_DISPLACEMENT_TAB_MODES),
            key="displacement_mode",
            horizontal=True,
        )

        if mode == "Осадки опор":
            render_support_settlements_table_and_force_method(solved_ss)
            st.info(
                "Для расчёта **Δ = Δ_P + Δ_c** по узлу ниже задайте единичное состояние и нажмите "
                "«Рассчитать единичное перемещение» — осадки из таблицы учитываются в **Δ_c**."
            )

        name_to_id = engine.node_name_to_model_node_id(solved_ss, st.session_state.nodes)
        available_nodes = sorted(name_to_id.keys())

        if mode == "Температурное (позже)":
            render_temperature_expansion_inputs()
            st.warning("Расчёт температурных перемещений в разработке.")
            if not available_nodes:
                st.warning("В расчётной модели не найдено узлов.")
        elif not available_nodes:
            st.warning("В расчетной модели не найдено узлов для запроса перемещения.")
        elif mode in (_DISPLACEMENT_TAB_MODES[0], _DISPLACEMENT_TAB_MODES[1]):
            if mode == "Только силовое (Метод Мора)":
                st.caption("Учитывается только **единичная нагрузка** (метод Мора / Верещагина). Осадки опор в расчёте **не** участвуют.")
            col_a, col_b, col_c = st.columns([1.2, 1.4, 1.0])
            with col_a:
                target_node = st.selectbox(
                    "Узел (единичное перемещение)",
                    options=available_nodes,
                    key="unit_target_node",
                )
            with col_b:
                direction_label = st.selectbox(
                    "Направление единичного перемещения",
                    options=["Горизонтальное (X)", "Вертикальное (Y)", "Поворот (Moment)"],
                    key="unit_target_direction",
                )
            with col_c:
                st.markdown("<br>", unsafe_allow_html=True)
                calc_unit = st.button("Рассчитать единичное перемещение", key="calc_unit_disp", use_container_width=True)

            direction_map = {
                "Горизонтальное (X)": "x",
                "Вертикальное (Y)": "y",
                "Поворот (Moment)": "rz",
            }
            selected_direction = direction_map[direction_label]

            st.markdown("##### Предпросмотр единичной нагрузки для заданного единичного перемещения")
            try:
                ss_unit_preview = engine.build_unit_preview_model(
                    solved_ss,
                    st.session_state.nodes,
                    target_node,
                    selected_direction,
                )
                fig_preview_unit = ss_unit_preview.show_structure(show=False, verbosity=1, offset=(0.0, 0.05))
                plotting.enhance_anastruct_figure(fig_preview_unit, arrow_color="#D32F2F", arrow_linewidth=3.2)
                plotting.overlay_all_session_nodes(fig_preview_unit)
                plotting.overlay_unit_load_marker(fig_preview_unit, target_node, selected_direction)
                st.pyplot(fig_preview_unit, clear_figure=True)
            except Exception as exc:
                st.warning(f"Не удалось построить предпросмотр для единичного перемещения: {exc}")

            if calc_unit:
                try:
                    ss_unit, disp, unit_solve_err = engine.calculate_unit_displacement(
                        solved_ss,
                        st.session_state.nodes,
                        target_node,
                        selected_direction,
                    )
                    if unit_solve_err:
                        st.error(unit_solve_err)
                        st.session_state.unit_result = None
                    else:
                        emap = st.session_state.get("session_to_ana_map")
                        if emap is None:
                            _, _, emap = engine.build_system(n, el, sup, ld, hn, global_ei=ei, global_ea=ea)
                        (
                            mohr_latex_lines,
                            mohr_formula,
                            mohr_compact_formula,
                            mohr_delta,
                            mohr_warnings,
                            mohr_text,
                            mohr_same_ei,
                            mohr_omega_y_sum,
                            mohr_ei_uniform,
                        ) = engine.build_vereshchagin_report(
                            solved_ss,
                            ss_unit,
                            emap,
                            st.session_state.nodes,
                            st.session_state.elements,
                        )
                        key = engine._displacement_component_key(selected_direction)
                        fe_value = float(disp[key])
                        error_abs = abs(fe_value - mohr_delta)
                        error_rel = (error_abs / (abs(fe_value) + 1e-12)) * 100.0

                        if mode == "Только силовое (Метод Мора)":
                            disp_for_settlement = _support_displacements_zeroed()
                        else:
                            sm.ensure_support_displacements_state()
                            disp_for_settlement = st.session_state.support_displacements

                        (
                            delta_settlement,
                            settlement_latex,
                            settlement_text,
                            settlement_warnings,
                            settlement_steps_latex,
                            settlement_formula_latex,
                            settlement_reaction_lines,
                            unit_reactions_latex,
                            compact_settlement_latex,
                            _settlement_sym_inner,
                            _settlement_num_inner,
                        ) = engine.compute_settlement_component(
                            ss_unit,
                            st.session_state.nodes,
                            st.session_state.supports,
                            disp_for_settlement,
                        )
                        delta_total = mohr_delta + delta_settlement
                        dtp = st.session_state.get("displacement_tab_params", {})

                        st.session_state.unit_result = {
                            "node": target_node,
                            "direction": direction_label,
                            "direction_code": selected_direction,
                            "displacement_mode": mode,
                            "disp": disp,
                            "disp_component_key": key,
                            "fe_value": fe_value,
                            "mohr_value": mohr_delta,
                            "mohr_latex_lines": mohr_latex_lines,
                            "mohr_formula": mohr_formula,
                            "mohr_compact_formula": mohr_compact_formula,
                            "mohr_text": mohr_text,
                            "mohr_warnings": mohr_warnings,
                            "mohr_same_ei": mohr_same_ei,
                            "mohr_omega_y_sum": mohr_omega_y_sum,
                            "mohr_ei_uniform": mohr_ei_uniform,
                            "delta_settlement": delta_settlement,
                            "settlement_latex": settlement_latex,
                            "settlement_text": settlement_text,
                            "settlement_warnings": settlement_warnings,
                            "settlement_steps_latex": settlement_steps_latex,
                            "settlement_formula_latex": settlement_formula_latex,
                            "settlement_reaction_lines": settlement_reaction_lines,
                            "unit_reactions_latex": unit_reactions_latex,
                            "compact_settlement_latex": compact_settlement_latex,
                            "thermal_tau_label": dtp.get("thermal_tau_label", "τ"),
                            "thermal_tau": float(dtp.get("thermal_tau", 0.0)),
                            "delta_total": delta_total,
                            "error_abs": error_abs,
                            "error_rel": error_rel,
                            "ss_unit": ss_unit,
                        }
                except Exception as exc:
                    st.error(f"Ошибка расчета единичного состояния: {exc}")
                    st.session_state.unit_result = None

            unit_result = st.session_state.unit_result
            if unit_result and (
                unit_result.get("node") != target_node
                or unit_result.get("direction_code") != selected_direction
                or unit_result.get("displacement_mode") != mode
            ):
                unit_result = None
            if unit_result:
                if unit_result["mohr_warnings"]:
                    st.warning("Предупреждения при интегрировании Мора:\n- " + "\n- ".join(unit_result["mohr_warnings"]))
                if unit_result["settlement_warnings"]:
                    st.warning("Предупреждения по осадкам опор:\n- " + "\n- ".join(unit_result["settlement_warnings"]))

                st.markdown("##### Грузовое состояние: реакции от внешних нагрузок")
                load_rows, _load_missed = engine.collect_support_reaction_rows(
                    solved_ss, st.session_state.nodes, st.session_state.supports
                )
                if load_rows:
                    for ltx in engine.format_load_reactions_latex(load_rows):
                        st.latex(ltx)
                else:
                    st.info("Не удалось сформировать реакции грузового состояния для узлов опор.")

                st.markdown("##### Подробный расчёт по методу Мора / Верещагина")
                for eq_line in engine.build_equilibrium_report(
                    solved_ss,
                    st.session_state.nodes,
                    st.session_state.elements,
                    st.session_state.supports,
                    st.session_state.loads,
                    st.session_state.hinges,
                ):
                    st.latex(eq_line)
                if unit_result["mohr_latex_lines"]:
                    for expr in unit_result["mohr_latex_lines"]:
                        st.latex(expr)
                    mc = unit_result["mohr_compact_formula"]
                    st.latex(mc)
                    ei_u = unit_result.get("mohr_ei_uniform")
                    wsum = float(unit_result.get("mohr_omega_y_sum", 0.0))
                    mv = float(unit_result["mohr_value"])
                    if (
                        unit_result.get("mohr_same_ei", True)
                        and ei_u is not None
                        and float(ei_u) > 1e-30
                    ):
                        st.latex(rf"= \frac{{{wsum:.8g}}}{{{float(ei_u):.8g}}} = {mv:.8g}")
                    else:
                        st.latex(rf"= {mv:.8g}")
                    st.latex(rf"\Delta_P = {mv:.4f}\,\text{{м}} = {mv * 1000.0:.1f}\,\text{{мм}}")
                    ds = float(unit_result.get("delta_settlement", 0.0))
                    show_settlement_walkthrough = unit_result.get("displacement_mode") == "Осадки опор"

                    if show_settlement_walkthrough:
                        st.markdown("##### Определение реакций опор от единичного воздействия")
                        st.markdown(
                            f"Задано **единичное перемещение**: узел **{unit_result.get('node', '')}**, "
                            f"направление **{unit_result.get('direction', '')}**. "
                            "Ниже — реакции опор **единичного состояния** "
                            r"($\overline{R}$, $\overline{M}$) в глобальных осях (направления указаны словами)."
                        )
                        for uline in unit_result.get("unit_reactions_latex") or []:
                            st.latex(uline)
                        _epu_u = plotting.epure_diagram_kwargs(
                            float(st.session_state.get("epure_scale", 0.65)),
                            values_deflection=1.0,
                        )
                        fig_m_unit_early = unit_result["ss_unit"].show_bending_moment(**_epu_u)
                        plotting.enhance_anastruct_figure(
                            fig_m_unit_early,
                            arrow_color="#D32F2F",
                            arrow_linewidth=2.4,
                            annotation_fontsize=10,
                        )
                        st.pyplot(fig_m_unit_early, clear_figure=True)
                        st.caption(
                            "Эпюра изгибающего момента **M̄** единичного состояния "
                            "(числовые подписи: verbosity=0 в параметрах эпюры)."
                        )

                        st.markdown("##### Расчёт перемещения от осадок опор")
                        comp_ltx = unit_result.get("compact_settlement_latex")
                        if comp_ltx:
                            st.latex(comp_ltx)
                    else:
                        st.markdown("##### Единичное состояние: реакции опор")
                        if unit_result.get("settlement_reaction_lines"):
                            for rline in unit_result["settlement_reaction_lines"]:
                                st.latex(rline)
                        else:
                            st.info("Нет заданных ненулевых осадок — блок реакций не выводится.")
                        st.markdown("##### Расчёт от осадок опор")
                        for step in unit_result.get("settlement_steps_latex", []):
                            st.latex(step)
                        st.latex(unit_result.get("settlement_formula_latex", unit_result["settlement_latex"]))
                        st.latex(r"\Delta_c = " + f"{ds:.8g}")
                        st.latex(rf"\Delta_c = {ds:.4f}\,\text{{м}} = {ds * 1000.0:.1f}\,\text{{мм}}")

                    dt = float(unit_result["delta_total"])
                    st.latex(_disp_delta_total_p_c_latex(mv, ds, dt))
                    st.markdown(f"`Δ_c` (численно): {unit_result['settlement_text']}")
                    tp = unit_result.get("thermal_tau", 0.0)
                    tl = unit_result.get("thermal_tau_label", "τ")
                    st.markdown(
                        f"Заданный температурный параметр **{tl}** = `{tp:.6e}` 1/°C "
                        "(в сумму ∫(M·m/EI) не входит, см. подпись выше)."
                    )
                else:
                    st.warning("Не удалось сформировать подробный расчёт по участкам.")

                st.download_button(
                    "Скачать текст расчёта (.txt)",
                    data=unit_result["mohr_text"],
                    file_name="mohr_report.txt",
                    mime="text/plain",
                    use_container_width=False,
                )

                st.markdown("##### Эпюры моментов: грузовое и единичное состояния")
                _epu = plotting.epure_diagram_kwargs(
                    float(st.session_state.get("epure_scale", 0.65)), values_deflection=1.0
                )
                uc1, uc2 = st.columns(2)
                with uc1:
                    fig_mg = solved_ss.show_bending_moment(**_epu)
                    plotting.enhance_anastruct_figure(
                        fig_mg, arrow_color="#D32F2F", arrow_linewidth=2.4, annotation_fontsize=10
                    )
                    st.pyplot(fig_mg, clear_figure=True)
                with uc2:
                    fig_mu = unit_result["ss_unit"].show_bending_moment(**_epu)
                    plotting.enhance_anastruct_figure(
                        fig_mu, arrow_color="#D32F2F", arrow_linewidth=2.4, annotation_fontsize=10
                    )
                    st.pyplot(fig_mu, clear_figure=True)

                st.markdown("##### Дополнительно: Q-эпюра единичного состояния")
                fig_uq = unit_result["ss_unit"].show_shear_force(**_epu)
                plotting.enhance_anastruct_figure(
                    fig_uq, arrow_color="#D32F2F", arrow_linewidth=2.4, annotation_fontsize=10
                )
                st.pyplot(fig_uq, clear_figure=True)
            else:
                st.info(
                    "Выберите узел и направление **единичного перемещения** и нажмите "
                    "«Рассчитать единичное перемещение», чтобы получить эпюры."
                )

    with tab_force:
        st.markdown("#### Метод сил")
        st.caption(
            "Канонические уравнения **Σ δ_ij·X_j + Δ_iF + Δ_is = 0** и их численное решение "
            "(**Δ_is** — свободные члены от осадок опор). "
            "Неизвестные **X₁, X₂, …** вводятся автоматически как реакции на единичные силы/моменты в узлах "
            "(учебная автоматизация; для серьёзного расчёта набор избыточных нужно выбирать вручную)."
        )
        sm.ensure_support_displacements_state()
        _n_fm, _el_fm, sup_fm, ld_fm, hn_fm, ei_fm, ea_fm = _editor_model_bundle()
        fm = engine.build_force_method_report(
            solved_ss,
            st.session_state.nodes,
            st.session_state.elements,
            sup_fm,
            hn_fm,
            ld_fm,
            max_unknowns=10,
            global_ei=ei_fm,
            global_ea=ea_fm,
            support_settlement_rows=list(st.session_state.get("support_settlement_rows") or []),
        )
        for w in fm.get("warnings", []):
            st.warning(w)
        if fm.get("tie_os_used"):
            st.success(
                "Метод сил: **основная система без затяжек**; неизвестные **Xᵢ** — усилия в разрезах: "
                + ", ".join(f"**{lab}**" for lab in fm.get("tie_labels_ru", []))
                + "."
            )
        for line in fm.get("markdown_intro", []):
            st.markdown(line)
        if fm.get("latex_lines"):
            st.markdown("**Канонические уравнения и суперпозиция (LaTeX):**")
            for lx in fm["latex_lines"]:
                st.latex(lx)
        for line in fm.get("markdown_solution", []):
            if line.strip() == "":
                st.markdown("")
            else:
                st.markdown(line)


if __name__ == "__main__":
    main()
