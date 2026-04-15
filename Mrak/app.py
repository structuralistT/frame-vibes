from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from anastruct import SystemElements
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


def configure_matplotlib_style() -> None:
    """Global matplotlib style for all charts in app."""
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["lines.linewidth"] = 2.5
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.5
    plt.rcParams["axes.grid"] = True
    plt.rcParams["figure.dpi"] = 110


def enhance_anastruct_figure(
    fig,
    *,
    arrow_color: str = "#D32F2F",
    arrow_linewidth: float = 2.4,
    text_offset_step: float = 0.015,
) -> None:
    """
    Improve readability of anaStruct-generated figures:
    - recolor + thicken arrows (loads/reactions),
    - apply consistent grid,
    - avoid overlap for texts with same position.
    """
    if not fig or not fig.axes:
        return

    ax = fig.axes[0]
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#12141a")
    ax.tick_params(colors="#ccc")
    ax.xaxis.label.set_color("#ccc")
    ax.yaxis.label.set_color("#ccc")
    if ax.get_title():
        ax.title.set_color("#ccc")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(True, linestyle="--", alpha=0.25, color="#555")

    # anaStruct рисует стержни через ax.plot(..., marker="s"); остальные линии (опоры, q-контур) не трогаем.
    _member_color = "#FFFFFF"
    for line in ax.lines:
        if isinstance(line, Line2D) and line.get_marker() == "s":
            line.set_color(_member_color)
            line.set_markerfacecolor(_member_color)
            line.set_markeredgecolor(_member_color)

    for patch in ax.patches:
        if isinstance(patch, FancyArrowPatch):
            patch.set_color(arrow_color)
            patch.set_linewidth(arrow_linewidth)
            try:
                patch.set_mutation_scale(max(12, patch.get_mutation_scale() * 1.2))
            except Exception:
                pass

    # Improve text readability and avoid complete overlap.
    seen_positions: dict[tuple[float, float], int] = {}
    y_span = max(abs(ax.get_ylim()[1] - ax.get_ylim()[0]), 1.0)
    for txt in ax.texts:
        txt.set_fontsize(14)
        txt.set_color("#FAFAFA")
        txt.set_bbox(dict(boxstyle="round,pad=0.18", facecolor="#1f242d", edgecolor="none", alpha=0.55))
        pos = txt.get_position()
        key = (round(float(pos[0]), 6), round(float(pos[1]), 6))
        idx = seen_positions.get(key, 0)
        if idx > 0:
            txt.set_position((pos[0], pos[1] + idx * text_offset_step * y_span))
        seen_positions[key] = idx + 1


def local_css(file_name: str) -> None:
    """Reads local CSS file and injects it into the app."""
    path = Path(file_name)
    if path.is_file():
        st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def init_state() -> None:
    """Initialize all session state containers used by the editor."""
    defaults = {
        "nodes": [],
        "elements": [],
        "supports": [],
        "loads": [],
        "hinges": [],
        "next_element_id": 1,
        "next_support_id": 1,
        "next_load_id": 1,
        "next_hinge_id": 1,
        "solved_ss": None,
        "solve_warnings": [],
        "unit_result": None,
        "support_displacements": {},
        "displacement_tab_params": {"thermal_tau": 0.0, "thermal_tau_label": "τ"},
        "global_EI": 5000.0,
        "global_EA": 1.0e9,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_node_by_name(name: str) -> dict | None:
    for node in st.session_state.nodes:
        if node["name"] == name:
            return node
    return None


def add_node(name: str, x: float, y: float) -> tuple[bool, str]:
    clean = name.strip()
    if not clean:
        return False, "Имя узла не может быть пустым."
    if get_node_by_name(clean):
        return False, f"Узел '{clean}' уже существует."
    st.session_state.nodes.append({"name": clean, "x": float(x), "y": float(y)})
    return True, f"Узел '{clean}' добавлен."


def delete_node(node_name: str) -> tuple[bool, str]:
    nodes = st.session_state.nodes
    if not any(n["name"] == node_name for n in nodes):
        return False, "Узел не найден."

    # Cascade delete: elements/supports/loads connected to the node.
    st.session_state.nodes = [n for n in nodes if n["name"] != node_name]
    removed_element_ids = {
        e["id"] for e in st.session_state.elements if e["start"] == node_name or e["end"] == node_name
    }
    st.session_state.elements = [
        e for e in st.session_state.elements if e["start"] != node_name and e["end"] != node_name
    ]
    st.session_state.hinges = [h for h in st.session_state.hinges if h.get("element_id") not in removed_element_ids]
    st.session_state.supports = [s for s in st.session_state.supports if s["node"] != node_name]
    st.session_state.loads = [
        l
        for l in st.session_state.loads
        if not (
            (l["type"] in ("point", "moment") and l.get("node") == node_name)
            or (l["type"] == "distributed" and l.get("element_id") in removed_element_ids)
        )
    ]
    return True, f"Узел '{node_name}' и связанные объекты удалены."


def add_element(start_node: str, end_node: str) -> tuple[bool, str]:
    if start_node == end_node:
        return False, "Начальный и конечный узлы должны отличаться."

    for element in st.session_state.elements:
        same_direct = element["start"] == start_node and element["end"] == end_node
        same_reverse = element["start"] == end_node and element["end"] == start_node
        if same_direct or same_reverse:
            return False, "Такой стержень уже существует."

    element_id = st.session_state.next_element_id
    st.session_state.next_element_id += 1
    st.session_state.elements.append({"id": element_id, "start": start_node, "end": end_node})
    return True, f"Стержень E{element_id} добавлен."


def delete_element(element_id: int) -> tuple[bool, str]:
    if not any(e["id"] == element_id for e in st.session_state.elements):
        return False, "Стержень не найден."

    st.session_state.elements = [e for e in st.session_state.elements if e["id"] != element_id]
    st.session_state.hinges = [h for h in st.session_state.hinges if int(h.get("element_id", -1)) != int(element_id)]
    st.session_state.loads = [
        l for l in st.session_state.loads if not (l["type"] == "distributed" and l.get("element_id") == element_id)
    ]
    return True, f"Стержень E{element_id} и связанные нагрузки удалены."


def add_support(node_name: str, support_type: str, angle_deg: float) -> tuple[bool, str]:
    # Keep one support per node for a clear model.
    st.session_state.supports = [s for s in st.session_state.supports if s["node"] != node_name]

    support_id = st.session_state.next_support_id
    st.session_state.next_support_id += 1
    st.session_state.supports.append(
        {"id": support_id, "node": node_name, "type": support_type, "angle": float(angle_deg)}
    )
    return True, f"Опора S{support_id} установлена на узел '{node_name}'."


def delete_support(support_id: int) -> tuple[bool, str]:
    if not any(s["id"] == support_id for s in st.session_state.supports):
        return False, "Опора не найдена."
    st.session_state.supports = [s for s in st.session_state.supports if s["id"] != support_id]
    return True, f"Опора S{support_id} удалена."


def add_load(load: dict) -> tuple[bool, str]:
    load_id = st.session_state.next_load_id
    st.session_state.next_load_id += 1
    payload = {"id": load_id, **load}
    st.session_state.loads.append(payload)
    return True, f"Нагрузка L{load_id} добавлена."


def delete_load(load_id: int) -> tuple[bool, str]:
    if not any(l["id"] == load_id for l in st.session_state.loads):
        return False, "Нагрузка не найдена."
    st.session_state.loads = [l for l in st.session_state.loads if l["id"] != load_id]
    return True, f"Нагрузка L{load_id} удалена."


def add_or_replace_hinge(element_id: int, position: str) -> tuple[bool, str]:
    """
    Add or replace an internal hinge on selected element end.
    position: "start" | "end"
    """
    if position not in ("start", "end"):
        return False, "Некорректное положение шарнира."

    element = next((e for e in st.session_state.elements if int(e["id"]) == int(element_id)), None)
    if not element:
        return False, "Выбранный стержень не найден."

    # Keep one hinge per element end.
    st.session_state.hinges = [
        h
        for h in st.session_state.hinges
        if not (int(h.get("element_id", -1)) == int(element_id) and h.get("position") == position)
    ]

    hinge_id = st.session_state.next_hinge_id
    st.session_state.next_hinge_id += 1
    st.session_state.hinges.append({"id": hinge_id, "element_id": int(element_id), "position": position})

    num_map = _element_number_by_id()
    el_num = num_map.get(int(element_id), int(element_id))
    pos_ru = "начало" if position == "start" else "конец"
    return True, f"Шарнир добавлен на стержень {el_num} ({pos_ru})."


def add_hinges_at_node(node_name: str) -> tuple[bool, str]:
    """
    Для каждого стержня, примыкающего к узлу node_name, добавить или заменить шарнир
    на соответствующем конце (start/end), вызывая add_or_replace_hinge.
    """
    incident: list[tuple[int, str]] = []
    for e in st.session_state.elements:
        eid = int(e["id"])
        if e.get("start") == node_name:
            incident.append((eid, "start"))
        elif e.get("end") == node_name:
            incident.append((eid, "end"))
    if not incident:
        return False, "Узел не принадлежит ни одному стержню"
    for eid, pos in incident:
        ok, msg = add_or_replace_hinge(eid, pos)
        if not ok:
            return False, msg
    n = len(incident)
    if n % 10 == 1 and n % 100 != 11:
        word = "стержень"
    elif 2 <= n % 10 <= 4 and (n % 100 < 10 or n % 100 >= 20):
        word = "стержня"
    else:
        word = "стержней"
    return True, f"Шарниры добавлены в узле {node_name} (затронуто {n} {word})"


def delete_hinge(hinge_id: int) -> tuple[bool, str]:
    if not any(int(h.get("id", -1)) == int(hinge_id) for h in st.session_state.hinges):
        return False, "Шарнир не найден."
    st.session_state.hinges = [h for h in st.session_state.hinges if int(h.get("id", -1)) != int(hinge_id)]
    return True, "Шарнир удален."


def delete_hinges_at_node(node_name: str) -> tuple[bool, str]:
    """Удалить все шарниры, привязанные к указанному узлу (по start/end стержня)."""
    before = len(st.session_state.hinges)
    st.session_state.hinges = [h for h in st.session_state.hinges if _hinge_node_name(h) != node_name]
    removed = before - len(st.session_state.hinges)
    if removed == 0:
        return False, f"В узле «{node_name}» нет шарниров."
    if removed % 10 == 1 and removed % 100 != 11:
        w = "шарнир"
    elif 2 <= removed % 10 <= 4 and (removed % 100 < 10 or removed % 100 >= 20):
        w = "шарнира"
    else:
        w = "шарниров"
    return True, f"Из узла «{node_name}» удалено {removed} {w}."


def delete_orphan_hinges() -> tuple[bool, str]:
    """Удалить шарниры, у которых не удаётся определить узел (например, стержень удалён)."""
    before = len(st.session_state.hinges)
    st.session_state.hinges = [h for h in st.session_state.hinges if _hinge_node_name(h) is not None]
    removed = before - len(st.session_state.hinges)
    if removed == 0:
        return False, "Таких шарниров нет."
    return True, f"Удалено шарниров без привязки к узлу: {removed}."


def build_system_from_state() -> tuple[SystemElements, list[str], dict[int, int]]:
    """
    Build an anastruct model from session state.
    Returns:
      - model
      - warnings list
      - map session element_id -> anastruct element_id
    """
    ss = SystemElements()
    warnings: list[str] = []

    if not st.session_state.elements:
        return ss, warnings, {}

    node_lookup = {n["name"]: n for n in st.session_state.nodes}
    session_to_anastruct_element: dict[int, int] = {}
    hinges_by_element: dict[int, list[str]] = {}
    for hinge in st.session_state.hinges:
        eid = int(hinge.get("element_id", -1))
        pos = hinge.get("position")
        if pos in ("start", "end"):
            hinges_by_element.setdefault(eid, []).append(pos)

    # 1) Elements
    for element in st.session_state.elements:
        n1 = node_lookup.get(element["start"])
        n2 = node_lookup.get(element["end"])
        if not n1 or not n2:
            warnings.append(f"E{element['id']}: пропущен, узлы не найдены.")
            continue
        if (n1["x"], n1["y"]) == (n2["x"], n2["y"]):
            warnings.append(f"E{element['id']}: пропущен, одинаковые координаты узлов.")
            continue

        anastruct_element_id = len(ss.element_map) + 1
        session_to_anastruct_element[element["id"]] = anastruct_element_id
        spring_dict: dict[int, float] = {}
        for pos in hinges_by_element.get(int(element["id"]), []):
            if pos == "start":
                spring_dict[1] = 0.0
            elif pos == "end":
                spring_dict[2] = 0.0
        ei_val = float(st.session_state.get("global_EI", 5000.0))
        ea_val = float(st.session_state.get("global_EA", 1.0e9))
        if spring_dict:
            ss.add_element(
                location=[[n1["x"], n1["y"]], [n2["x"], n2["y"]]],
                EA=ea_val,
                EI=ei_val,
                spring=spring_dict,
            )
        else:
            ss.add_element(
                location=[[n1["x"], n1["y"]], [n2["x"], n2["y"]]],
                EA=ea_val,
                EI=ei_val,
            )

    # 2) Node mapping by coordinates after elements are created.
    coord_to_node_id = {
        (round(node.vertex.x, 9), round(node.vertex.y, 9)): node_id for node_id, node in ss.node_map.items()
    }
    name_to_node_id: dict[str, int] = {}
    for node in st.session_state.nodes:
        key = (round(node["x"], 9), round(node["y"], 9))
        if key in coord_to_node_id:
            name_to_node_id[node["name"]] = coord_to_node_id[key]

    # 3) Supports
    for support in st.session_state.supports:
        node_id = name_to_node_id.get(support["node"])
        if not node_id:
            warnings.append(f"S{support['id']}: узел '{support['node']}' не входит в текущие стержни.")
            continue

        if support["type"] == "fixed":
            ss.add_support_fixed(node_id=node_id)
        elif support["type"] == "hinged":
            ss.add_support_hinged(node_id=node_id)
        elif support["type"] == "roller":
            ss.add_support_roll(node_id=node_id, angle=support["angle"])

    # 4) Loads
    for load in st.session_state.loads:
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
            anastruct_element_id = session_to_anastruct_element.get(load["element_id"])
            if anastruct_element_id:
                ss.q_load(q=load["q"], element_id=anastruct_element_id)
            else:
                warnings.append(f"L{load['id']}: распределенная нагрузка пропущена (стержень вне схемы).")

    return ss, warnings, session_to_anastruct_element


def _preview_axis_limits() -> tuple[float, float, float, float]:
    """Padding around all node coordinates for consistent axes."""
    nodes = st.session_state.nodes
    if not nodes:
        return -1.0, 1.0, -1.0, 1.0
    xs = [float(n["x"]) for n in nodes]
    ys = [float(n["y"]) for n in nodes]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    pad = 0.18 * span
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def _node_lookup() -> dict[str, dict]:
    return {n["name"]: n for n in st.session_state.nodes}


def overlay_all_session_nodes(fig, *, color: str = "#F39C12") -> None:
    """Draw every user node on top of anaStruct figure (names always visible)."""
    if not fig.axes or not st.session_state.nodes:
        return
    ax = fig.axes[0]
    for n in st.session_state.nodes:
        ax.plot(
            n["x"],
            n["y"],
            "o",
            ms=10,
            color=color,
            zorder=22,
            markeredgecolor="white",
            markeredgewidth=2.0,
        )
        ax.annotate(
            n["name"],
            (n["x"], n["y"]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=13,
            fontweight="bold",
            color="#FAFAFA",
            zorder=24,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#262730", edgecolor=color, alpha=0.92),
        )


def overlay_unit_load_marker(fig, node_name: str, unit_direction: str) -> None:
    """Add explicit red marker arrow showing where unit action is applied."""
    if not fig or not fig.axes:
        return
    node = get_node_by_name(node_name)
    if not node:
        return

    ax = fig.axes[0]
    x, y = float(node["x"]), float(node["y"])
    span = max(abs(ax.get_xlim()[1] - ax.get_xlim()[0]), abs(ax.get_ylim()[1] - ax.get_ylim()[0]), 1.0)
    d = 0.08 * span

    if unit_direction == "x":
        ax.annotate(
            "",
            xy=(x + d, y),
            xytext=(x, y),
            arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=3.2),
            zorder=30,
        )
    elif unit_direction == "y":
        ax.annotate(
            "",
            xy=(x, y - d),
            xytext=(x, y),
            arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=3.2),
            zorder=30,
        )
    elif unit_direction == "rz":
        ax.text(
            x + 0.03 * span,
            y + 0.03 * span,
            "↻",
            color="#D32F2F",
            fontsize=20,
            fontweight="bold",
            zorder=31,
        )


def _hinge_node_name(hinge: dict) -> str | None:
    """Return node name where hinge is attached (based on element end)."""
    el = next((e for e in st.session_state.elements if int(e["id"]) == int(hinge.get("element_id", -1))), None)
    if not el:
        return None
    if hinge.get("position") == "start":
        return el["start"]
    if hinge.get("position") == "end":
        return el["end"]
    return None


def overlay_session_hinges(fig) -> None:
    """Draw hinge markers from session state on top of preview figure."""
    if not fig or not fig.axes or not st.session_state.hinges:
        return
    ax = fig.axes[0]
    node_lookup = {n["name"]: n for n in st.session_state.nodes}
    span = max(abs(ax.get_xlim()[1] - ax.get_xlim()[0]), abs(ax.get_ylim()[1] - ax.get_ylim()[0]), 1.0)
    # Радиус порядка видимого размера узла (ms≈10) на типичном масштабе осей.
    r = 0.007 * span

    for hinge in st.session_state.hinges:
        node_name = _hinge_node_name(hinge)
        node = node_lookup.get(node_name) if node_name else None
        if not node:
            continue
        circ = plt.Circle(
            (float(node["x"]), float(node["y"])),
            r,
            facecolor="#0E1117",
            edgecolor="#FAFAFA",
            linewidth=1.4,
            zorder=28,
        )
        ax.add_patch(circ)


def draw_standalone_live_preview() -> plt.Figure:
    """
    Full live preview without anaStruct geometry (e.g. no bars yet):
    nodes, member lines, simple support/load glyphs — all from session_state.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    nodes = st.session_state.nodes
    lookup = _node_lookup()
    xmin, xmax, ymin, ymax = _preview_axis_limits()
    span = max(xmax - xmin, ymax - ymin, 1e-6)
    glyph = span * 0.035

    if not nodes:
        ax.text(0.5, 0.5, "Добавьте узлы — схема появится здесь.", ha="center", va="center", fontsize=13, color="#888")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#0E1117")
        return fig

    # Members
    for el in st.session_state.elements:
        a, b = lookup.get(el["start"]), lookup.get(el["end"])
        if not a or not b:
            continue
        ax.plot([a["x"], b["x"]], [a["y"], b["y"]], color="#FFFFFF", lw=2.5, zorder=1, solid_capstyle="round")

    # Supports (simple symbols at node)
    for sup in st.session_state.supports:
        nd = lookup.get(sup["node"])
        if not nd:
            continue
        x, y = nd["x"], nd["y"]
        if sup["type"] == "fixed":
            tri = plt.Polygon(
                [[x, y], [x - glyph, y - 1.4 * glyph], [x + glyph, y - 1.4 * glyph]],
                closed=True,
                color="#7F8C8D",
                zorder=3,
            )
            ax.add_patch(tri)
        elif sup["type"] == "hinged":
            tri = plt.Polygon(
                [[x, y], [x - 0.85 * glyph, y - 1.2 * glyph], [x + 0.85 * glyph, y - 1.2 * glyph]],
                closed=True,
                fill=False,
                edgecolor="#ECF0F1",
                lw=2,
                zorder=3,
            )
            ax.add_patch(tri)
        else:
            ang = np.radians(sup.get("angle", 0.0))
            ux, uy = np.cos(ang), np.sin(ang)
            roll_len = glyph * 1.6
            ax.plot(
                [x - ux * roll_len * 0.5, x + ux * roll_len * 0.5],
                [y - uy * roll_len * 0.5, y + uy * roll_len * 0.5],
                color="#ECF0F1",
                lw=2,
                zorder=2,
            )
            circ = plt.Circle((x, y), glyph * 0.35, fill=True, facecolor="#F39C12", edgecolor="white", zorder=4)
            ax.add_patch(circ)

    # Loads
    scale = span * 0.08
    for load in st.session_state.loads:
        if load["type"] == "point":
            nd = lookup.get(load["node"])
            if not nd:
                continue
            fx, fy = float(load.get("Fx", 0)), float(load.get("Fy", 0))
            if fx == 0 and fy == 0:
                continue
            ax.annotate(
                "",
                xytext=(nd["x"], nd["y"]),
                xy=(nd["x"] + fx * scale * 0.15, nd["y"] + fy * scale * 0.15),
                arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=2),
                zorder=8,
            )
        elif load["type"] == "moment":
            nd = lookup.get(load["node"])
            if not nd:
                continue
            m = float(load.get("M", 0))
            ax.annotate(f"M={m:g}", (nd["x"], nd["y"]), xytext=(0, -22), textcoords="offset points", ha="center", color="#9B59B6", fontsize=10, fontweight="bold", zorder=8)
        elif load["type"] == "distributed":
            el = next((e for e in st.session_state.elements if e["id"] == load["element_id"]), None)
            if not el:
                continue
            a, b = lookup.get(el["start"]), lookup.get(el["end"])
            if not a or not b:
                continue
            q = float(load.get("q", 0))
            mx, my = (a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2
            dx, dy = b["x"] - a["x"], b["y"] - a["y"]
            ln = (dx * dx + dy * dy) ** 0.5 or 1.0
            px, py = -dy / ln, dx / ln
            off = span * 0.04 * (1 if q <= 0 else -1)
            ax.annotate(f"q={q:g}", (mx + px * off, my + py * off), ha="center", fontsize=9, color="#3498DB", zorder=8)

    # Nodes on top (above members, supports, load arrows — zorder >= 20)
    for n in nodes:
        ax.plot(
            n["x"],
            n["y"],
            "o",
            ms=10,
            color="#F39C12",
            zorder=22,
            markeredgecolor="white",
            markeredgewidth=2.0,
        )
        ax.annotate(
            n["name"],
            (n["x"], n["y"]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=13,
            fontweight="bold",
            color="#FAFAFA",
            zorder=24,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#262730", edgecolor="#F39C12", alpha=0.95),
        )

    # Internal hinges from session state (always visible in preview).
    for hinge in st.session_state.hinges:
        node_name = _hinge_node_name(hinge)
        nd = lookup.get(node_name) if node_name else None
        if not nd:
            continue
        circ = plt.Circle(
            (float(nd["x"]), float(nd["y"])),
            glyph * 0.24,
            facecolor="#0E1117",
            edgecolor="#FAFAFA",
            linewidth=1.4,
            zorder=12,
        )
        ax.add_patch(circ)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2, color="#555")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#12141a")
    ax.tick_params(colors="#ccc")
    ax.xaxis.label.set_color("#ccc")
    ax.yaxis.label.set_color("#ccc")
    for spine in ax.spines.values():
        spine.set_color("#444")
    return fig


def render_live_preview(preview_ss: SystemElements) -> None:
    """
    Всегда показываем актуальную модель из session_state.
    Если в anaStruct есть стержни — show_structure + поверх все узлы с именами;
    иначе — полностью своя отрисовка (в т.ч. только узлы / «висячие» узлы).
    """
    has_bars_in_model = bool(preview_ss.element_map)
    if has_bars_in_model:
        try:
            # IMPORTANT: verbosity=0 keeps load glyphs visible (point/q/moment loads).
            fig = preview_ss.show_structure(show=False, verbosity=0, offset=(0.0, 0.05))
            enhance_anastruct_figure(fig, arrow_color="#D32F2F", arrow_linewidth=2.6)
            if st.session_state.nodes:
                ax_pv = fig.axes[0]
                pxmin, pxmax, pymin, pymax = _preview_axis_limits()
                cxmin, cxmax = ax_pv.get_xlim()
                cymin, cymax = ax_pv.get_ylim()
                ax_pv.set_xlim(min(cxmin, pxmin), max(cxmax, pxmax))
                ax_pv.set_ylim(min(cymin, pymin), max(cymax, pymax))
            overlay_all_session_nodes(fig)
            overlay_session_hinges(fig)
            st.pyplot(fig, clear_figure=True)
        except Exception as exc:
            st.error(f"Ошибка при построении предварительной схемы: {exc}")
            fig = draw_standalone_live_preview()
            overlay_session_hinges(fig)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
    else:
        fig = draw_standalone_live_preview()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)


def format_load_reactions_latex(reaction_rows: list[dict]) -> list[str]:
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


def collect_support_reaction_rows(ss: SystemElements) -> tuple[list[dict], list[str]]:
    """Collect support reactions as rows for tables/cards."""
    if not st.session_state.supports:
        return [], []

    coord_map = _coord_to_model_node_id(ss)
    node_by_name = {n["name"]: n for n in st.session_state.nodes}

    reaction_rows: list[dict] = []
    missed: list[str] = []
    for support in st.session_state.supports:
        name = support["node"]
        node_data = node_by_name.get(name)
        if not node_data:
            missed.append(f"Узел '{name}' не найден в исходных данных.")
            continue

        key = (round(float(node_data["x"]), 9), round(float(node_data["y"]), 9))
        model_node_id = coord_map.get(key)
        if not model_node_id:
            missed.append(f"Узел '{name}' не попал в расчетную схему.")
            continue

        node = ss.node_map[model_node_id]
        symbol = "".join(ch for ch in name.lower() if ch.isalnum()) or "n"
        reaction_rows.append(
            {
                "Узел": name,
                "Индекс": symbol,
                "Тип опоры": _support_type_label(support["type"]),
                "Тип опоры (код)": support["type"],
                "Fx": float(node.Fx),
                "Fy": float(node.Fy),
                "Mz (Tz)": float(node.Tz),
                "Направление Fx": "вправо" if float(node.Fx) >= 0 else "влево",
                "Направление Fy": "вверх" if float(node.Fy) >= 0 else "вниз",
                "Направление Mz": "против часовой" if float(node.Tz) >= 0 else "по часовой",
            }
        )
    return reaction_rows, missed


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


def _coord_to_model_node_id(ss: SystemElements) -> dict[tuple[float, float], int]:
    return {(round(node.vertex.x, 9), round(node.vertex.y, 9)): node_id for node_id, node in ss.node_map.items()}


def _node_name_to_model_node_id(ss: SystemElements) -> dict[str, int]:
    coord_map = _coord_to_model_node_id(ss)
    name_to_id: dict[str, int] = {}
    for node in st.session_state.nodes:
        key = (round(float(node["x"]), 9), round(float(node["y"]), 9))
        node_id = coord_map.get(key)
        if node_id is not None:
            name_to_id[node["name"]] = node_id
    return name_to_id


def _support_type_label(s_type: str) -> str:
    labels = {
        "fixed": "Fixed (жесткая заделка)",
        "hinged": "Hinged (шарнирно-неподвижная)",
        "roller": "Roller (шарнирно-подвижная)",
    }
    return labels.get(s_type, s_type)


def _fmt_num(v: float) -> str:
    return f"{float(v):.1f}"


def _force_arrow(fx: float, fy: float) -> str:
    # Priority: dominant component; fallback to combined marker.
    if abs(fx) >= abs(fy) and abs(fx) > 1e-12:
        return "→" if fx > 0 else "←"
    if abs(fy) > 1e-12:
        return "↑" if fy > 0 else "↓"
    return "•"


def _element_label_for_load(element_id: int) -> str:
    element = next((e for e in st.session_state.elements if e["id"] == element_id), None)
    if not element:
        return f"E{element_id}"
    return f"{element['start']}{element['end']}"


def _element_number_by_id() -> dict[int, int]:
    """
    UI numbering for elements: 1..N in current order.
    Recomputed each rerun, so numbering stays compact after deletions.
    """
    return {int(e["id"]): i for i, e in enumerate(st.session_state.elements, start=1)}


def _element_ui_label(element: dict) -> str:
    num_map = _element_number_by_id()
    num = num_map.get(int(element["id"]), int(element["id"]))
    return f"Стержень {num} ({element['start']} → {element['end']})"


def _hinge_ui_label(hinge: dict) -> str:
    element = next((e for e in st.session_state.elements if int(e["id"]) == int(hinge["element_id"])), None)
    if not element:
        el_text = f"Стержень {hinge['element_id']}"
    else:
        el_text = _element_ui_label(element)
    pos_ru = "начало" if hinge.get("position") == "start" else "конец"
    return f"{el_text}, {pos_ru}"


def format_load_short(load: dict) -> str:
    """Human-friendly compact load label for UI."""
    if load["type"] == "point":
        node = load["node"]
        fx = float(load.get("Fx", 0.0))
        fy = float(load.get("Fy", 0.0))
        if abs(fx) > 1e-12 and abs(fy) > 1e-12:
            x_arrow = "→" if fx > 0 else "←"
            y_arrow = "↑" if fy > 0 else "↓"
            return f"F_{node} = ({_fmt_num(abs(fx))} {x_arrow}, {_fmt_num(abs(fy))} {y_arrow}) кН"
        mag = abs(fx) if abs(fx) > 1e-12 else abs(fy)
        return f"F_{node} = {_fmt_num(mag)} кН ({_force_arrow(fx, fy)})"

    if load["type"] == "distributed":
        num_map = _element_number_by_id()
        el_num = num_map.get(int(load["element_id"]), int(load["element_id"]))
        q = float(load.get("q", 0.0))
        q_arrow = "↓" if q < 0 else "↑"
        return f"q_{el_num} = {_fmt_num(abs(q))} кН/м ({q_arrow})"

    # moment
    node = load["node"]
    m = float(load.get("M", 0.0))
    return f"M_{node} = {_fmt_num(abs(m))} кН·м (↻)"


def render_loads_summary() -> None:
    """Compact summary of user-defined loads."""
    if not st.session_state.loads:
        st.info("Заданные нагрузки: пока нет.")
        return

    lines: list[str] = []
    for load in st.session_state.loads:
        lines.append(f"- {format_load_short(load)}")

    st.markdown("#### Заданные нагрузки")
    st.info("\n".join(lines))


def count_independent_cycles_in_bar_graph() -> int:
    """
    Количество независимых замкнутых контуров K по графу стержней (узлы — имена узлов,
    рёбра — стержни). Для каждой связной компоненты: K_c = max(0, E_c - V_c + 1)
    (цикломатическое число; параллельные стержни между одной парой узлов учитываются как отдельные рёбра).

    Полная классическая оценка степени статической неопределимости рам с жёсткими узлами
    и произвольной топологией требует учёта шарнирных узлов, составных шарниров и т.д.;
    здесь K — только топологический цикл по «жёсткой» схеме стержней из редактора.
    """
    elements = st.session_state.elements
    if not elements:
        return 0
    node_names = {str(n["name"]) for n in st.session_state.nodes}
    adj: dict[str, set[str]] = {}
    edges: list[tuple[str, str]] = []
    for el in elements:
        a, b = str(el.get("start", "")), str(el.get("end", ""))
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


def calculate_static_indeterminacy_n() -> tuple[int, int, dict[str, int], int]:
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
    support_counts = {"fixed": 0, "hinged": 0, "roller": 0}
    for support in st.session_state.supports:
        s_type = support.get("type")
        if s_type in support_counts:
            support_counts[s_type] += 1

    sop = 3 * support_counts["fixed"] + 2 * support_counts["hinged"] + 1 * support_counts["roller"]
    # Internal hinges reduce static indeterminacy.
    # For each node with k hinged member ends, reduction = (k - 1).
    node_hinge_counts: dict[str, int] = {}
    element_by_id = {int(e["id"]): e for e in st.session_state.elements}
    for h in st.session_state.hinges:
        eid = int(h.get("element_id", -1))
        pos = h.get("position")
        el = element_by_id.get(eid)
        if not el:
            continue
        if pos == "start":
            node_name = el["start"]
        elif pos == "end":
            node_name = el["end"]
        else:
            continue
        node_hinge_counts[node_name] = node_hinge_counts.get(node_name, 0) + 1

    hinge_relief = sum(max(0, k - 1) for k in node_hinge_counts.values())
    n_simple = sop - 3 - hinge_relief

    k_loops = count_independent_cycles_in_bar_graph()
    sh_total = len(st.session_state.hinges)
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


def build_equilibrium_report(ss_load: SystemElements) -> list[str]:
    """
    LaTeX-строки глобальных уравнений равновесия плоской системы от внешних нагрузок и реакций опор.
    Для n ≠ 0 — краткое сообщение про МКЭ (рукописная развёртка не формируется).
    """
    n_val, _, _, _ = calculate_static_indeterminacy_n()
    if n_val != 0:
        return [r"\text{Реакции определены методом конечных элементов (МКЭ).}"]

    if not st.session_state.supports:
        return [r"\text{Нет опор — уравнения равновесия не формулируются.}"]

    node_xy = {str(n["name"]): (float(n["x"]), float(n["y"])) for n in st.session_state.nodes}
    elem_by_id = {int(e["id"]): e for e in st.session_state.elements}

    ref = str(st.session_state.supports[0]["node"])
    if ref not in node_xy:
        return [r"\text{Не удалось сопоставить первую опору с узлом для уравнения моментов.}"]
    xa, ya = node_xy[ref]

    ext_fx = 0.0
    ext_fy = 0.0
    ext_m_a = 0.0

    for load in st.session_state.loads:
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
            eid = int(load.get("element_id", -1))
            el = elem_by_id.get(eid)
            if not el:
                continue
            na, nb = str(el["start"]), str(el["end"])
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

    reaction_rows, _miss = collect_support_reaction_rows(ss_load)
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
    nodes_count: int,
    elements_count: int,
) -> None:
    """
    Подробное пояснение для случая n < 0: схема ведёт себя как механизм (нехватка связей).
    Логика расчёта n_val не меняется — только вывод.
    """
    W = -int(n_val)
    n_hinges = len(st.session_state.hinges)
    w_check = 3 * int(nodes_count) - 2 * int(elements_count) - int(sop)

    st.warning("Схема **мгновенно изменяема**: конструкция ведёт себя как **механизм**, а не как неподвижная система.")

    st.markdown(
        f"По оценке из редактора степень статической неопределимости **n = {n_val}**. "
        f"Отрицательное значение n означает, что **не хватает** внешних или внутренних связей, "
        f"чтобы схема устойчиво стояла на месте."
    )

    st.markdown(
        f"**Количество степеней свободы механизма** (в смысле этой оценки): **W = −n = {W}**. "
        f"Это значит, что схема может совершать **{_russian_degrees_of_freedom_phrase(W)}** "
        f"без растяжения и сжатия материала — то есть «шевелиться» как недостаточно закреплённый каркас."
    )

    if W == 1:
        st.info(
            "При **одной** степени свободы обычно можно представить одно возможное малое перемещение "
            "(например, поворот или сдвиг целиком), пока не добавите опоры или жёсткие узлы."
        )
    else:
        st.info(
            f"У схемы **{_russian_degrees_of_freedom_phrase(W)}** — возможны несколько независимых малых перемещений; "
            f"такую конструкцию в расчёте опорных усилий нельзя считать заданной по форме."
        )

    st.markdown(
        "**Проверка по простой «школьной» формуле для плоской стержневой схемы** "
        "(узлы как шарниры, стержни как стержни, **внутренние шарниры в формулу не входят** — это только первый ориентир):"
    )
    st.markdown(
        r"$$W_{\mathrm{пр}} = 3U - 2S - S_{\mathrm{оп}}$$"
        f" — здесь **U** = число узлов (**{nodes_count}**), **S** = число стержней (**{elements_count}**), "
        f"**S_оп** = число опорных связей (**{sop}**): жёсткая заделка даёт 3 связи, шарнирно-неподвижная — 2, катящаяся — 1."
    )
    st.info(
        f"Подставляем: **W_пр = 3·{nodes_count} − 2·{elements_count} − {sop} = {w_check}**. "
        f"Число внутренних шарниров в модели сейчас: **{n_hinges}** — они уменьшают жёсткость, но в этой короткой формуле не учтены, "
        f"поэтому **W_пр** может не совпадать с **W = {W}** из показателя **n**; расхождение как раз намекает, что нужно учитывать шарниры и реальную схему опирания."
    )

    st.markdown(
        f"Для справки: по графу стержней **K = {k_loops}** замкнутых контуров; опоры: жёстких **{support_counts.get('fixed', 0)}**, "
        f"шарнирно-неподвижных **{support_counts.get('hinged', 0)}**, катящихся **{support_counts.get('roller', 0)}**."
    )

    if sop > 3 and n_val < 0:
        st.warning(
            "Опорных связей **больше трёх**, но схема всё равно «шатается». Так бывает, если связи **не удерживают все возможные движения** "
            "(например, несколько катящихся опор в одну линию, нет устойчивости «в сторону», или опоры стоят так, что всё равно можно повернуть целиком). "
            "Имеет смысл проверить расположение опор и жёсткость узлов."
        )


def render_static_determinacy_block() -> None:
    """Compact status line for static determinacy."""
    n_val, sop, cnt, k_loops = calculate_static_indeterminacy_n()
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
        render_kinematic_analysis(
            n_val,
            sop,
            cnt,
            k_loops,
            len(st.session_state.nodes),
            len(st.session_state.elements),
        )


def calculate_unit_displacement(base_ss: SystemElements, node_name: str, unit_direction: str) -> tuple[SystemElements, dict]:
    """
    Unit-state analysis (Mohr-method equivalent via FE displacement extraction):
    - deepcopy solved model
    - remove all real loads
    - apply unit load/moment in selected node
    - solve and return displacement dict for this node
    """
    name_to_id = _node_name_to_model_node_id(base_ss)
    if node_name not in name_to_id:
        raise ValueError("Выбранный узел не найден в расчетной модели.")

    node_id = name_to_id[node_name]
    ss_unit = copy.deepcopy(base_ss)
    ss_unit.remove_loads()

    if unit_direction == "x":
        ss_unit.point_load(node_id=node_id, Fx=1.0, Fy=0.0)
    elif unit_direction == "y":
        # Standard sign convention for vertical unit load: downward.
        ss_unit.point_load(node_id=node_id, Fx=0.0, Fy=-1.0)
    elif unit_direction == "rz":
        ss_unit.moment_load(node_id=node_id, Tz=1.0)
    else:
        raise ValueError("Неизвестное направление единичной нагрузки.")

    ss_unit.solve()
    disp = ss_unit.get_node_displacements(node_id)
    return ss_unit, disp


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


def _migrate_support_displacement_entry(support: dict, entry: dict) -> dict:
    """Ensure symbolic labels exist for stored support displacement dict."""
    node = str(support["node"])
    s_type = support["type"]
    out = dict(entry)
    if s_type == "roller":
        if "sym_dn" not in out or not str(out.get("sym_dn", "")).strip():
            out["sym_dn"] = f"c_{node}n"
        if "dn_mm" not in out:
            dx = float(out.get("dx_mm", 0.0))
            dy = float(out.get("dy_mm", 0.0))
            out["dn_mm"] = float(np.hypot(dx, dy))
    else:
        if "sym_dx" not in out or not str(out.get("sym_dx", "")).strip():
            out["sym_dx"] = f"c_{node}x"
        if "sym_dy" not in out or not str(out.get("sym_dy", "")).strip():
            out["sym_dy"] = f"c_{node}y"
        if s_type == "fixed":
            if "sym_phi" not in out or not str(out.get("sym_phi", "")).strip():
                out["sym_phi"] = f"φ_{node}"
    return out


def ensure_support_displacements_state() -> None:
    """Keep support displacement inputs synchronized with existing supports."""
    sd = st.session_state.support_displacements
    active_ids = {int(s["id"]) for s in st.session_state.supports}

    # Remove stale entries
    for sid in list(sd.keys()):
        if int(sid) not in active_ids:
            sd.pop(sid, None)

    # Add defaults for new supports
    for support in st.session_state.supports:
        sid = int(support["id"])
        node = str(support["node"])
        s_type = support["type"]
        if sid not in sd:
            if s_type == "roller":
                sd[sid] = {
                    "dn_mm": 0.0,
                    "dx_mm": 0.0,
                    "dy_mm": 0.0,
                    "phi_rad": 0.0,
                    "sym_dn": f"c_{node}n",
                }
            elif s_type == "hinged":
                sd[sid] = {
                    "dx_mm": 0.0,
                    "dy_mm": 0.0,
                    "phi_rad": 0.0,
                    "sym_dx": f"c_{node}x",
                    "sym_dy": f"c_{node}y",
                }
            else:
                sd[sid] = {
                    "dx_mm": 0.0,
                    "dy_mm": 0.0,
                    "phi_rad": 0.0,
                    "sym_dx": f"c_{node}x",
                    "sym_dy": f"c_{node}y",
                    "sym_phi": f"φ_{node}",
                }
        else:
            sd[sid] = _migrate_support_displacement_entry(support, sd[sid])


def render_support_displacements_inputs() -> None:
    """UI block for support settlements/displacements (symbolic labels + values)."""
    st.markdown("##### Смещения опор")
    if not st.session_state.supports:
        st.info("Опоры не заданы.")
        return

    ensure_support_displacements_state()
    for support in st.session_state.supports:
        sid = int(support["id"])
        node = support["node"]
        s_type = support["type"]
        label = f"S{sid} ({node}, {s_type})"
        st.markdown(f"**{label}**")

        current = st.session_state.support_displacements[sid]
        if s_type == "roller":
            angle_deg = float(support.get("angle", 90.0))
            c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.0, 1.0])
            with c1:
                sym_dn = st.text_input(
                    "Обозначение смещения",
                    value=str(current.get("sym_dn", f"c_{node}n")),
                    key=f"sup_sym_dn_{sid}",
                    help="Символ для смещения по нормали к связи (как в учебнике).",
                )
            with c2:
                dn_mm = st.number_input(
                    "Значение, мм",
                    value=float(current.get("dn_mm", 0.0)),
                    step=0.1,
                    key=f"sup_dn_{sid}",
                )
            with c3:
                st.caption(f"Нормаль: {angle_deg:.1f}° к X")
            with c4:
                st.caption("Проекции на X/Y считаются автоматически.")

            a = np.radians(angle_deg)
            dx = float(dn_mm) * float(np.cos(a))
            dy = float(dn_mm) * float(np.sin(a))
            st.session_state.support_displacements[sid] = {
                "sym_dn": str(sym_dn).strip() or f"c_{node}n",
                "dn_mm": float(dn_mm),
                "dx_mm": dx,
                "dy_mm": dy,
                "phi_rad": 0.0,
            }
        elif s_type == "hinged":
            r1a, r1b = st.columns(2)
            r2a, r2b = st.columns(2)
            with r1a:
                sym_dx = st.text_input(
                    "Горизонтальное смещение — обозначение",
                    value=str(current.get("sym_dx", f"c_{node}x")),
                    key=f"sup_sym_dx_{sid}",
                )
            with r1b:
                dx = st.number_input(
                    "мм",
                    value=float(current.get("dx_mm", 0.0)),
                    step=0.1,
                    key=f"sup_dx_{sid}",
                )
            with r2a:
                sym_dy = st.text_input(
                    "Вертикальное смещение — обозначение",
                    value=str(current.get("sym_dy", f"c_{node}y")),
                    key=f"sup_sym_dy_{sid}",
                )
            with r2b:
                dy = st.number_input(
                    "мм",
                    value=float(current.get("dy_mm", 0.0)),
                    step=0.1,
                    key=f"sup_dy_{sid}",
                )
            st.session_state.support_displacements[sid] = {
                "sym_dx": str(sym_dx).strip() or f"c_{node}x",
                "sym_dy": str(sym_dy).strip() or f"c_{node}y",
                "dx_mm": float(dx),
                "dy_mm": float(dy),
                "phi_rad": 0.0,
            }
        else:  # fixed
            r1a, r1b = st.columns(2)
            r2a, r2b = st.columns(2)
            r3a, r3b = st.columns(2)
            with r1a:
                sym_dx = st.text_input(
                    "Δx — обозначение",
                    value=str(current.get("sym_dx", f"c_{node}x")),
                    key=f"sup_sym_dx_{sid}",
                )
            with r1b:
                dx = st.number_input(
                    "мм",
                    value=float(current.get("dx_mm", 0.0)),
                    step=0.1,
                    key=f"sup_dx_{sid}",
                )
            with r2a:
                sym_dy = st.text_input(
                    "Δy — обозначение",
                    value=str(current.get("sym_dy", f"c_{node}y")),
                    key=f"sup_sym_dy_{sid}",
                )
            with r2b:
                dy = st.number_input(
                    "мм",
                    value=float(current.get("dy_mm", 0.0)),
                    step=0.1,
                    key=f"sup_dy_{sid}",
                )
            with r3a:
                sym_phi = st.text_input(
                    "Угол поворота — обозначение",
                    value=str(current.get("sym_phi", f"φ_{node}")),
                    key=f"sup_sym_phi_{sid}",
                )
            with r3b:
                phi = st.number_input(
                    "рад",
                    value=float(current.get("phi_rad", 0.0)),
                    step=0.0001,
                    key=f"sup_phi_{sid}",
                )
            st.session_state.support_displacements[sid] = {
                "sym_dx": str(sym_dx).strip() or f"c_{node}x",
                "sym_dy": str(sym_dy).strip() or f"c_{node}y",
                "sym_phi": str(sym_phi).strip() or f"φ_{node}",
                "dx_mm": float(dx),
                "dy_mm": float(dy),
                "phi_rad": float(phi),
            }


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


def compute_settlement_component(ss_unit: SystemElements) -> tuple[float, str, str, list[str], list[str], str, list[str]]:
    """
    Compute settlement contribution:
      Δ_c = - Σ (R_i^(1) · Δ_i)
    where R_i^(1) are support reactions from the unit (virtual) state and Δ_i are prescribed settlements.

    Returns an extra list of LaTeX lines summarizing unit-state reactions at supports with settlements.
    """
    ensure_support_displacements_state()
    name_to_id = _node_name_to_model_node_id(ss_unit)

    term_values: list[float] = []
    term_outer_parts: list[str] = []
    term_text_parts: list[str] = []
    term_numeric_parts: list[str] = []
    settlement_steps_latex: list[str] = []
    reaction_summary_lines: list[str] = []
    warnings: list[str] = []

    eps_mm = 1e-9
    eps_rad = 1e-12

    def support_has_settlement_input(support: dict, disp: dict) -> bool:
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

    for support in st.session_state.supports:
        sid = int(support["id"])
        node_name = support["node"]
        node_id = name_to_id.get(node_name)
        if node_id is None:
            warnings.append(f"S{sid}: узел '{node_name}' не найден в единичной схеме.")
            continue

        disp = st.session_state.support_displacements.get(sid, {})
        dx_m = float(disp.get("dx_mm", 0.0)) / 1000.0
        dy_m = float(disp.get("dy_mm", 0.0)) / 1000.0
        phi = float(disp.get("phi_rad", 0.0))
        s_type = support["type"]

        sym_dx = str(disp.get("sym_dx", f"c_{node_name}x")).strip() or f"c_{node_name}x"
        sym_dy = str(disp.get("sym_dy", f"c_{node_name}y")).strip() or f"c_{node_name}y"
        sym_phi = str(disp.get("sym_phi", f"φ_{node_name}")).strip() or f"φ_{node_name}"
        sym_dn = str(disp.get("sym_dn", f"c_{node_name}n")).strip() or f"c_{node_name}n"
        lx_dx = _user_sym_to_latex(sym_dx)
        lx_dy = _user_sym_to_latex(sym_dy)
        lx_phi = _user_sym_to_latex(sym_phi)
        lx_dn = _user_sym_to_latex(sym_dn)

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
            Rn = rf"\overline{{R}}_{{\text{{{node_name}n}}}}"
            term_outer_parts.append(rf"{Rn}\,{lx_dn}")
            term_text_parts.append(f"{R_eq:.6g}·{dn_m:.6g}")
            term_numeric_parts.append(f"{val:.6g}")
            settlement_steps_latex.append(
                rf"{Rn}\cdot {lx_dn} = {R_eq:.6g}\cdot {dn_m:.6g} = {val:.6g}"
            )
            continue

        if abs(dx_m) > 1e-15:
            val = rx * dx_m
            term_values.append(val)
            Rbx = rf"\overline{{R}}_{{{node_name}x}}"
            term_outer_parts.append(rf"{Rbx}\,{lx_dx}")
            term_text_parts.append(f"{rx:.6g}·{dx_m:.6g}")
            term_numeric_parts.append(f"{val:.6g}")
            settlement_steps_latex.append(
                rf"{Rbx}\cdot {lx_dx} = {rx:.6g}\cdot {dx_m:.6g} = {val:.6g}"
            )
        if abs(dy_m) > 1e-15:
            val = ry * dy_m
            term_values.append(val)
            Rby = rf"\overline{{R}}_{{{node_name}y}}"
            term_outer_parts.append(rf"{Rby}\,{lx_dy}")
            term_text_parts.append(f"{ry:.6g}·{dy_m:.6g}")
            term_numeric_parts.append(f"{val:.6g}")
            settlement_steps_latex.append(
                rf"{Rby}\cdot {lx_dy} = {ry:.6g}\cdot {dy_m:.6g} = {val:.6g}"
            )
        if s_type == "fixed" and abs(phi) > 1e-15:
            val = rm * phi
            term_values.append(val)
            Mb = rf"\overline{{M}}_{{{node_name}}}"
            term_outer_parts.append(rf"{Mb}\,{lx_phi}")
            term_text_parts.append(f"{rm:.6g}·{phi:.6g}")
            term_numeric_parts.append(f"{val:.6g}")
            settlement_steps_latex.append(
                rf"{Mb}\cdot {lx_phi} = {rm:.6g}\cdot {phi:.6g} = {val:.6g}"
            )

    for support in st.session_state.supports:
        sid = int(support["id"])
        node_name = support["node"]
        node_id = name_to_id.get(node_name)
        if node_id is None:
            continue
        disp = st.session_state.support_displacements.get(sid, {})
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
        return 0.0, r"\Delta_{c} = 0", "0", warnings, [], r"\Delta_{c}=0", reaction_summary_lines

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
        + f"{delta_c:.8g}"
    )
    text_expr = "-(" + " + ".join(term_text_parts) + f") = {delta_c:.8g}"
    return delta_c, latex_expr, text_expr, warnings, settlement_steps_latex, latex_formula, reaction_summary_lines


def _displacement_component_key(unit_direction: str) -> str:
    return {"x": "ux", "y": "uy", "rz": "phi_z"}[unit_direction]


def _format_component_name(unit_direction: str) -> str:
    return {"x": "Δx", "y": "Δy", "rz": "φ"}[unit_direction]


def _format_component_value(unit_direction: str, value: float) -> str:
    if unit_direction in ("x", "y"):
        return f"{value * 1000:.3f} мм"
    return f"{value:.6f} рад"


def compute_mohr_displacement_report(
    ss_load: SystemElements, ss_unit: SystemElements
) -> tuple[float, list[dict], list[str]]:
    """
    Compute displacement using Mohr integral:
      delta = sum( integral(M * m dx) / EI )
    using numerical integration along each element.
    """
    rows: list[dict] = []
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
    ss_load: SystemElements, ss_unit: SystemElements
) -> tuple[list[str], str, str, float, list[str], str, bool, float, float | None]:
    _, _, session_to_ana = build_system_from_state()
    ana_to_session = {int(v): int(k) for k, v in session_to_ana.items()}
    node_xy = _node_lookup()

    def n(v: float) -> str:
        s = f"{float(v):.4f}".rstrip("0").rstrip(".")
        return s if s else "0"

    def _close_xy(a: tuple[float, float], b: tuple[float, float], tol: float = 1e-5) -> bool:
        return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol

    def section_label(element_id: int, e_load) -> str:
        sess_id = ana_to_session.get(int(element_id))
        element = None
        if sess_id is not None:
            element = next((e for e in st.session_state.elements if int(e["id"]) == int(sess_id)), None)
        if not element:
            element = next((e for e in st.session_state.elements if int(e["id"]) == int(element_id)), None)
        if element:
            return f"{element['start']}{element['end']}"
        x1, y1 = float(e_load.vertex_1.x), float(e_load.vertex_1.y)
        x2, y2 = float(e_load.vertex_2.x), float(e_load.vertex_2.y)
        for se in st.session_state.elements:
            na = node_xy.get(se["start"])
            nb = node_xy.get(se["end"])
            if not na or not nb:
                continue
            p1 = (float(na["x"]), float(na["y"]))
            p2 = (float(nb["x"]), float(nb["y"]))
            if (_close_xy((x1, y1), p1) and _close_xy((x2, y2), p2)) or (_close_xy((x1, y1), p2) and _close_xy((x2, y2), p1)):
                return f"{se['start']}{se['end']}"
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


def build_unit_preview_model(base_ss: SystemElements, node_name: str, unit_direction: str) -> SystemElements:
    """Create temporary model for preview of a single unit action."""
    name_to_id = _node_name_to_model_node_id(base_ss)
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
    base_solved: SystemElements, node_name: str, unit_direction: str
) -> SystemElements:
    """Модель с единичным воздействием и выполненным solve (эпюры M̄ для метода сил)."""
    ss = build_unit_preview_model(base_solved, node_name, unit_direction)
    ss.solve()
    return ss


def _fmt_canonical_coef(v: float) -> str:
    x = float(v)
    if abs(x) < 1e-14:
        return "0"
    s = f"{x:.4g}"
    if "e" in s.lower():
        s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def build_force_method_report(
    ss_load_solved: SystemElements,
    *,
    max_unknowns: int = 10,
) -> dict:
    """
    Канонические уравнения метода сил в виде LaTeX-строк и численное решение.

    Принимается стандартная форма совместности по перемещениям:
      \\sum_{j=1}^{n} \\delta_{ij} X_j + \\Delta_{iF} = 0,
    где \\delta_{ij} = \\int (\\bar M_i \\bar M_j / \\mathrm{EI})\\,\\mathrm{d}x,
    \\Delta_{iF} = \\int (M_F \\bar M_i / \\mathrm{EI})\\,\\mathrm{d}x — через тот же численный интеграл Мора,
    что и в ``compute_mohr_displacement_report``.

    Избыточные неизвестные X_1,\\dots,X_n задаются автоматически как реакции на **единичные**
    силы/моменты в узлах (перебор узлов и направлений Y, X, момент). Для реальных задач набор
    избыточных нужно выбирать осмысленно; здесь — учебный автоматический вариант при небольшом n.
    """
    markdown_intro: list[str] = []
    latex_lines: list[str] = []
    markdown_solution: list[str] = []
    warnings: list[str] = []

    n_target, _sop, _cnt, _k = calculate_static_indeterminacy_n()
    if int(n_target) <= 0:
        return {
            "ok": False,
            "markdown_intro": [
                "**Метод сил** в канонической форме используется, когда степень статической "
                "неопределимости **n > 0**. Сейчас **n ≤ 0** — для этой схемы блок канонических уравнений не строится "
                "(см. сообщение о статической определимости выше)."
            ],
            "latex_lines": [],
            "markdown_solution": [],
            "warnings": [],
        }

    name_to_id = _node_name_to_model_node_id(ss_load_solved)
    available = sorted(name_to_id.keys())
    if not available:
        return {
            "ok": False,
            "markdown_intro": ["Нет узлов в расчётной модели — метод сил не применим."],
            "latex_lines": [],
            "markdown_solution": [],
            "warnings": [],
        }

    pairs: list[tuple[str, str]] = []
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
            ssu = _solved_unit_system_for_force_method(ss_load_solved, node, d)
        except Exception as exc:
            return {
                "ok": False,
                "markdown_intro": [f"Не удалось построить единичное состояние **{k + 1}**: `{exc}`."],
                "latex_lines": [],
                "markdown_solution": [],
                "warnings": warnings + [str(exc)],
            }
        unit_states.append(ssu)
        unit_labels_ru.append(f"узел **{node}**, {dir_ru[d]}")

    n = n_req
    markdown_intro.append(
        f"Степень статической неопределимости (по оценке редактора): **n = {n_target}**. "
        f"Вводим **{n}** избыточных неизвестных **X₁ … X_{n}** (силы или моменты в «лишних» связях в учебной постановке)."
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

    latex_lines.append(
        rf"\text{{Неизвестные:}}\quad X_1,\,X_2,\,\ldots,\,X_{{{n}}}\quad"
        rf"\text{{ (силы в кН или моменты в кН·м в зависимости от единицы).}}"
    )
    latex_lines.append(
        rf"\sum_{{j=1}}^{{{n}}} \delta_{{ij}}\, X_j + \Delta_{{iF}} = 0,\qquad i=1,\ldots,{n}."
    )
    latex_lines.append(r"\text{Численные значения коэффициентов (интеграл Мора):}")
    for i in range(n):
        parts = [rf"{_fmt_canonical_coef(delta[i, j])} \cdot X_{{{j + 1}}}" for j in range(n)]
        parts.append(_fmt_canonical_coef(DeltaF[i]))
        latex_lines.append(" + ".join(parts) + r" = 0")

    latex_lines.append(r"\text{или в матричном виде:}\quad [\delta]\,\mathbf{X} = -\mathbf{\Delta}_F")

    try:
        X = np.linalg.solve(delta, -DeltaF)
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
    return {
        "ok": True,
        "markdown_intro": markdown_intro,
        "latex_lines": latex_lines,
        "markdown_solution": markdown_solution,
        "warnings": uniq_warn,
        "n_used": n,
        "n_target": int(n_target),
        "X": X.tolist(),
        "cond": cond,
    }


def render_support_reactions(ss: SystemElements) -> None:
    """Render support reactions table and reaction-force plot."""
    st.markdown("#### Реакции опор")
    if not st.session_state.supports:
        st.info("Опоры не заданы. Реакции отсутствуют.")
        return

    reaction_rows, missed = collect_support_reaction_rows(ss)

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
        enhance_anastruct_figure(fig_react, arrow_color="#D32F2F", arrow_linewidth=2.6)
        st.pyplot(fig_react, clear_figure=True)
    except Exception as exc:
        st.error(f"Не удалось построить график реакций: {exc}")


def render_sidebar() -> bool:
    with st.sidebar:
        st.markdown("## Редактор модели")

        # Nodes
        with st.expander("1) Узлы", expanded=True):
            with st.form("add_node_form", clear_on_submit=True):
                node_name = st.text_input("Имя узла", placeholder="A")
                col1, col2 = st.columns(2)
                with col1:
                    x = st.number_input("X", value=0.0, step=1.0, key="node_x")
                with col2:
                    y = st.number_input("Y", value=0.0, step=1.0, key="node_y")
                if st.form_submit_button("Добавить узел"):
                    ok, msg = add_node(node_name, x, y)
                    (st.success if ok else st.error)(msg)

            if st.session_state.nodes:
                node_options = [n["name"] for n in st.session_state.nodes]
                node_to_delete = st.selectbox("Удалить узел", options=node_options, key="node_to_delete")
                if st.button("Удалить выбранный узел"):
                    ok, msg = delete_node(node_to_delete)
                    (st.success if ok else st.error)(msg)
                st.dataframe(st.session_state.nodes, use_container_width=True, hide_index=True)
            else:
                st.info("Узлы пока не добавлены.")

        # Elements
        with st.expander("2) Стержни", expanded=True):
            node_names = [n["name"] for n in st.session_state.nodes]
            if len(node_names) >= 2:
                with st.form("add_element_form", clear_on_submit=True):
                    start = st.selectbox("Начальный узел", options=node_names, key="elem_start")
                    end = st.selectbox("Конечный узел", options=node_names, key="elem_end")
                    if st.form_submit_button("Добавить стержень"):
                        ok, msg = add_element(start, end)
                        (st.success if ok else st.error)(msg)
            else:
                st.info("Добавьте минимум 2 узла.")

            if st.session_state.elements:
                element_ids = [int(e["id"]) for e in st.session_state.elements]
                by_id = {int(e["id"]): e for e in st.session_state.elements}
                label = st.selectbox(
                    "Удалить стержень",
                    options=element_ids,
                    format_func=lambda eid: _element_ui_label(by_id[eid]),
                    key="element_to_delete",
                )
                if st.button("Удалить выбранный стержень"):
                    ok, msg = delete_element(label)
                    (st.success if ok else st.error)(msg)
                st.dataframe(st.session_state.elements, use_container_width=True, hide_index=True)
            else:
                st.info("Стержни пока не добавлены.")

        # Supports
        with st.expander("3) Опоры", expanded=True):
            node_names = [n["name"] for n in st.session_state.nodes]
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
                    angle = st.number_input(
                        "Угол Roller, град (0 = ось X)",
                        value=0.0,
                        step=5.0,
                        key="support_angle",
                        disabled=support_type != "roller",
                    )
                    if st.form_submit_button("Добавить/заменить опору"):
                        ok, msg = add_support(node, support_type, angle)
                        (st.success if ok else st.error)(msg)
            else:
                st.info("Сначала добавьте узлы.")

            if st.session_state.supports:
                support_labels = [
                    f"S{s['id']}: {s['node']} [{s['type']}{', angle=' + str(s['angle']) if s['type']=='roller' else ''}]"
                    for s in st.session_state.supports
                ]
                label = st.selectbox("Удалить опору", options=support_labels, key="support_to_delete")
                if st.button("Удалить выбранную опору"):
                    support_id = int(label.split(":")[0][1:])
                    ok, msg = delete_support(support_id)
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

            node_names = [n["name"] for n in st.session_state.nodes]
            if load_kind == "point":
                if node_names:
                    with st.form("add_point_load_form", clear_on_submit=True):
                        node = st.selectbox("Узел", options=node_names, key="point_node")
                        fx = st.number_input("Fx", value=0.0, step=1.0, key="point_fx")
                        fy = st.number_input("Fy", value=0.0, step=1.0, key="point_fy")
                        if st.form_submit_button("Добавить нагрузку"):
                            ok, msg = add_load({"type": "point", "node": node, "Fx": float(fx), "Fy": float(fy)})
                            (st.success if ok else st.error)(msg)
                else:
                    st.info("Нет узлов для задания нагрузки.")

            elif load_kind == "distributed":
                if st.session_state.elements:
                    with st.form("add_distributed_load_form", clear_on_submit=True):
                        element_ids = [int(e["id"]) for e in st.session_state.elements]
                        by_id = {int(e["id"]): e for e in st.session_state.elements}
                        picked_id = st.selectbox(
                            "Стержень",
                            options=element_ids,
                            format_func=lambda eid: _element_ui_label(by_id[eid]),
                            key="q_element",
                        )
                        q = st.number_input("q (отрицательное = вниз)", value=-5.0, step=1.0, key="q_value")
                        if st.form_submit_button("Добавить нагрузку"):
                            ok, msg = add_load({"type": "distributed", "element_id": int(picked_id), "q": float(q)})
                            (st.success if ok else st.error)(msg)
                else:
                    st.info("Нет стержней для задания распределенной нагрузки.")

            elif load_kind == "moment":
                if node_names:
                    with st.form("add_moment_load_form", clear_on_submit=True):
                        node = st.selectbox("Узел", options=node_names, key="moment_node")
                        mz = st.number_input("Момент Mz", value=5.0, step=1.0, key="moment_value")
                        if st.form_submit_button("Добавить нагрузку"):
                            ok, msg = add_load({"type": "moment", "node": node, "M": float(mz)})
                            (st.success if ok else st.error)(msg)
                else:
                    st.info("Нет узлов для задания момента.")

            if st.session_state.loads:
                load_ids = [int(load["id"]) for load in st.session_state.loads]
                labels_by_id = {int(load["id"]): format_load_short(load) for load in st.session_state.loads}
                selected_load_id = st.selectbox(
                    "Удалить нагрузку",
                    options=load_ids,
                    format_func=lambda lid: labels_by_id.get(lid, f"Нагрузка {lid}"),
                    key="load_to_delete",
                )
                if st.button("Удалить выбранную нагрузку"):
                    ok, msg = delete_load(selected_load_id)
                    (st.success if ok else st.error)(msg)
                st.dataframe(st.session_state.loads, use_container_width=True, hide_index=True)
            else:
                st.info("Нагрузки пока не добавлены.")

        # Hinges
        with st.expander("5) Шарниры", expanded=False):
            if st.session_state.elements:
                node_names = [n["name"] for n in st.session_state.nodes]
                if node_names:
                    with st.form("add_hinge_form", clear_on_submit=True):
                        hinge_node = st.selectbox("Узел", options=node_names, key="hinge_node_name")
                        if st.form_submit_button("Добавить шарниры в узле"):
                            ok, msg = add_hinges_at_node(hinge_node)
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
                    nn = _hinge_node_name(h)
                    if nn is None:
                        orphans.append(h)
                    else:
                        by_node[nn].append(h)
                for node_nm in sorted(by_node.keys()):
                    hinges_here = by_node[node_nm]
                    row_l, row_r = st.columns([3, 2], gap="small")
                    with row_l:
                        detail = " · ".join(_hinge_ui_label(x) for x in hinges_here)
                        st.markdown(f"**Узел {node_nm}** ({len(hinges_here)} шт.)")
                        st.caption(detail)
                    with row_r:
                        if st.button(
                            "Удалить все в узле",
                            key=f"delete_hinges_node_{node_nm}",
                            use_container_width=True,
                        ):
                            ok, msg = delete_hinges_at_node(node_nm)
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
                            ok, msg = delete_orphan_hinges()
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

        st.markdown("### Сводка по нагрузкам")
        render_loads_summary()
        st.markdown("---")
        return st.button("Узнать горькую правду", type="primary", use_container_width=True)

def main() -> None:
    st.set_page_config(page_title="Plane Frame Solver", page_icon="📐", layout="wide", initial_sidebar_state="expanded")
    configure_matplotlib_style()
    st.markdown(
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">',
        unsafe_allow_html=True,
    )
    local_css("style.css")
    init_state()

    st.markdown('<div class="app-title">Редактор плоских рам (Streamlit + anaStruct)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Твои дома обязательно будут стоять.</div>',
        unsafe_allow_html=True,
    )

    calculate_pressed = render_sidebar()

    # Live preview (без solve): узлы, стержни, опоры и нагрузки — всё из session_state
    preview_ss, preview_warnings, _ = build_system_from_state()
    st.markdown("### Предварительная схема (live)")
    render_static_determinacy_block()
    render_live_preview(preview_ss)

    if preview_warnings:
        st.warning("Некоторые объекты не отображены в превью:\n- " + "\n- ".join(preview_warnings))

    if calculate_pressed:
        if not st.session_state.elements:
            st.error("Расчет невозможен: нет стержней.")
        elif not st.session_state.supports:
            st.error("Расчет невозможен: не задано ни одной опоры.")
        else:
            try:
                solved_ss, solve_warnings, _ = build_system_from_state()
                solved_ss.solve()
                st.session_state.solved_ss = solved_ss
                st.session_state.solve_warnings = solve_warnings
                st.session_state.unit_result = None
            except Exception as exc:
                st.error(f"Ошибка расчета: {exc}")
                st.session_state.solved_ss = None
                st.session_state.solve_warnings = []

    solved_ss = st.session_state.solved_ss
    if solved_ss is None:
        return

    st.markdown("### Результаты расчета")
    if st.session_state.solve_warnings:
        st.warning("Некоторые объекты были пропущены при расчете:\n- " + "\n- ".join(st.session_state.solve_warnings))

    tab_eff, tab_disp, tab_force = st.tabs(["Усилия и реакции", "Перемещения", "Метод сил"])

    with tab_eff:
        reaction_rows, reaction_missed = collect_support_reaction_rows(solved_ss)
        render_static_determinacy_block()
        result_card(solved_ss, reaction_rows)
        if reaction_missed:
            st.warning("Часть реакций не удалось сопоставить:\n- " + "\n- ".join(reaction_missed))

        st.markdown("#### Эпюры M, Q, N")
        c1, c2, c3 = st.columns(3)
        with c1:
            fig_m = solved_ss.show_bending_moment(show=False, verbosity=1, offset=(0.0, 0.06))
            enhance_anastruct_figure(fig_m, arrow_color="#D32F2F", arrow_linewidth=2.4)
            st.pyplot(fig_m, clear_figure=True)
        with c2:
            fig_q = solved_ss.show_shear_force(show=False, verbosity=1, offset=(0.0, 0.06))
            enhance_anastruct_figure(fig_q, arrow_color="#D32F2F", arrow_linewidth=2.4)
            st.pyplot(fig_q, clear_figure=True)
        with c3:
            fig_n = solved_ss.show_axial_force(show=False, verbosity=1, offset=(0.0, 0.06))
            enhance_anastruct_figure(fig_n, arrow_color="#D32F2F", arrow_linewidth=2.4)
            st.pyplot(fig_n, clear_figure=True)

        render_support_reactions(solved_ss)

    with tab_disp:
        st.markdown("#### Перемещения (единичное состояние, метод Мора)")
        st.caption("Ниже задайте **единичное перемещение** (узел и направление или поворот).")
        render_support_displacements_inputs()
        render_temperature_expansion_inputs()
        name_to_id = _node_name_to_model_node_id(solved_ss)
        available_nodes = sorted(name_to_id.keys())

        if not available_nodes:
            st.warning("В расчетной модели не найдено узлов для запроса перемещения.")
        else:
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
                ss_unit_preview = build_unit_preview_model(
                    solved_ss,
                    node_name=target_node,
                    unit_direction=selected_direction,
                )
                fig_preview_unit = ss_unit_preview.show_structure(show=False, verbosity=1, offset=(0.0, 0.05))
                enhance_anastruct_figure(fig_preview_unit, arrow_color="#D32F2F", arrow_linewidth=3.2)
                overlay_all_session_nodes(fig_preview_unit)
                overlay_unit_load_marker(fig_preview_unit, target_node, selected_direction)
                st.pyplot(fig_preview_unit, clear_figure=True)
            except Exception as exc:
                st.warning(f"Не удалось построить предпросмотр для единичного перемещения: {exc}")

            if calc_unit:
                try:
                    ss_unit, disp = calculate_unit_displacement(
                        solved_ss,
                        node_name=target_node,
                        unit_direction=selected_direction,
                    )
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
                    ) = build_vereshchagin_report(solved_ss, ss_unit)
                    key = _displacement_component_key(selected_direction)
                    fe_value = float(disp[key])
                    error_abs = abs(fe_value - mohr_delta)
                    error_rel = (error_abs / (abs(fe_value) + 1e-12)) * 100.0
                    (
                        delta_settlement,
                        settlement_latex,
                        settlement_text,
                        settlement_warnings,
                        settlement_steps_latex,
                        settlement_formula_latex,
                        settlement_reaction_lines,
                    ) = compute_settlement_component(ss_unit)
                    delta_total = mohr_delta + delta_settlement
                    dtp = st.session_state.get("displacement_tab_params", {})

                    st.session_state.unit_result = {
                        "node": target_node,
                        "direction": direction_label,
                        "direction_code": selected_direction,
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
                unit_result.get("node") != target_node or unit_result.get("direction_code") != selected_direction
            ):
                unit_result = None
            if unit_result:
                if unit_result["mohr_warnings"]:
                    st.warning("Предупреждения при интегрировании Мора:\n- " + "\n- ".join(unit_result["mohr_warnings"]))
                if unit_result["settlement_warnings"]:
                    st.warning("Предупреждения по осадкам опор:\n- " + "\n- ".join(unit_result["settlement_warnings"]))

                st.markdown("##### Грузовое состояние: реакции от внешних нагрузок")
                load_rows, _load_missed = collect_support_reaction_rows(solved_ss)
                if load_rows:
                    for ltx in format_load_reactions_latex(load_rows):
                        st.latex(ltx)
                else:
                    st.info("Не удалось сформировать реакции грузового состояния для узлов опор.")

                st.markdown("##### Подробный расчёт по методу Мора / Верещагина")
                for eq_line in build_equilibrium_report(solved_ss):
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
                    ds = float(unit_result["delta_settlement"])
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
                uc1, uc2 = st.columns(2)
                with uc1:
                    fig_mg = solved_ss.show_bending_moment(show=False, verbosity=1, offset=(0.0, 0.06))
                    enhance_anastruct_figure(fig_mg, arrow_color="#D32F2F", arrow_linewidth=2.4)
                    st.pyplot(fig_mg, clear_figure=True)
                with uc2:
                    fig_mu = unit_result["ss_unit"].show_bending_moment(show=False, verbosity=1, offset=(0.0, 0.06))
                    enhance_anastruct_figure(fig_mu, arrow_color="#D32F2F", arrow_linewidth=2.4)
                    st.pyplot(fig_mu, clear_figure=True)

                st.markdown("##### Дополнительно: Q-эпюра единичного состояния")
                fig_uq = unit_result["ss_unit"].show_shear_force(show=False, verbosity=1, offset=(0.0, 0.06))
                enhance_anastruct_figure(fig_uq, arrow_color="#D32F2F", arrow_linewidth=2.4)
                st.pyplot(fig_uq, clear_figure=True)
            else:
                st.info(
                    "Выберите узел и направление **единичного перемещения** и нажмите "
                    "«Рассчитать единичное перемещение», чтобы получить эпюры."
                )

    with tab_force:
        st.markdown("#### Метод сил")
        st.caption(
            "Канонические уравнения **Σ δ_ij·X_j + Δ_iF = 0** и их численное решение. "
            "Неизвестные **X₁, X₂, …** вводятся автоматически как реакции на единичные силы/моменты в узлах "
            "(учебная автоматизация; для серьёзного расчёта набор избыточных нужно выбирать вручную)."
        )
        fm = build_force_method_report(solved_ss)
        for w in fm.get("warnings", []):
            st.warning(w)
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
