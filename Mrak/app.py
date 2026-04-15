from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from anastruct import SystemElements
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
    ax.grid(True, linestyle="--", alpha=0.5)

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
        "next_element_id": 1,
        "next_support_id": 1,
        "next_load_id": 1,
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
        ss.add_element(location=[[n1["x"], n1["y"]], [n2["x"], n2["y"]]])

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
            zorder=15,
            markeredgecolor="white",
            markeredgewidth=1.0,
        )
        ax.annotate(
            n["name"],
            (n["x"], n["y"]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color="#FAFAFA",
            zorder=16,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#262730", edgecolor=color, alpha=0.92),
        )


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
        ax.plot([a["x"], b["x"]], [a["y"], b["y"]], color="#BDC3C7", lw=2.5, zorder=1, solid_capstyle="round")

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

    # Nodes on top
    for n in nodes:
        ax.plot(
            n["x"],
            n["y"],
            "o",
            ms=11,
            color="#F39C12",
            zorder=10,
            markeredgecolor="white",
            markeredgewidth=1.2,
        )
        ax.annotate(
            n["name"],
            (n["x"], n["y"]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color="#FAFAFA",
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#262730", edgecolor="#F39C12", alpha=0.95),
        )

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
            fig = preview_ss.show_structure(show=False, verbosity=1, offset=(0.0, 0.05))
            enhance_anastruct_figure(fig, arrow_color="#D32F2F", arrow_linewidth=2.6)
            overlay_all_session_nodes(fig)
            st.pyplot(fig, clear_figure=True)
        except Exception as exc:
            st.error(f"Ошибка при построении предварительной схемы: {exc}")
            fig = draw_standalone_live_preview()
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
    else:
        fig = draw_standalone_live_preview()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)


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


def _support_type_label(s_type: str) -> str:
    labels = {
        "fixed": "Fixed (жесткая заделка)",
        "hinged": "Hinged (шарнирно-неподвижная)",
        "roller": "Roller (шарнирно-подвижная)",
    }
    return labels.get(s_type, s_type)


def render_loads_summary() -> None:
    """Compact summary of user-defined loads."""
    if not st.session_state.loads:
        st.info("Заданные нагрузки: пока нет.")
        return

    lines: list[str] = []
    for load in st.session_state.loads:
        if load["type"] == "point":
            lines.append(f"- L{load['id']}: Point @ {load['node']} -> Fx={load['Fx']:.3f} кН, Fy={load['Fy']:.3f} кН")
        elif load["type"] == "distributed":
            lines.append(f"- L{load['id']}: Distributed @ E{load['element_id']} -> q={load['q']:.3f} кН/м")
        else:
            lines.append(f"- L{load['id']}: Moment @ {load['node']} -> M={load['M']:.3f} кН·м")

    st.markdown("#### Заданные нагрузки")
    st.info("\n".join(lines))


def calculate_static_indeterminacy_n() -> tuple[int, int, dict[str, int]]:
    """
    Static indeterminacy for single-disk planar frame (no internal hinges):
      n = Sop - 3
    where Sop is total number of support constraints.
    """
    support_counts = {"fixed": 0, "hinged": 0, "roller": 0}
    for support in st.session_state.supports:
        s_type = support.get("type")
        if s_type in support_counts:
            support_counts[s_type] += 1

    sop = 3 * support_counts["fixed"] + 2 * support_counts["hinged"] + 1 * support_counts["roller"]
    n_val = sop - 3
    return n_val, sop, support_counts


def render_static_determinacy_block() -> None:
    """Show static indeterminacy n based on support constraints."""
    n_val, sop, cnt = calculate_static_indeterminacy_n()
    breakdown = f"Соп = 3*{cnt['fixed']} (fixed) + 2*{cnt['hinged']} (hinged) + 1*{cnt['roller']} (roller) = {sop}"
    formula = f"n = Соп - 3 = {sop} - 3 = {n_val}"

    st.info(breakdown)
    if n_val == 0:
        st.success(f"{formula} -> Система статически определима.")
    elif n_val > 0:
        st.info(f"{formula} -> Система статически неопределима, степень n = {n_val}.")
    else:
        st.warning(f"{formula} -> Система мгновенно изменяема (механизм), не хватает {-n_val} связей.")

    st.caption("Используется модель единого жесткого диска: roller всегда дает 1 связь.")
    st.caption("Для общей схемы с несколькими дисками: n = 3*Д - 2*Ш - Соп.")
    # Future extension example (if internal hinges are introduced):
    # D = number_of_disks
    # Sh = number_of_simple_internal_hinges
    # n_general = 3 * D - 2 * Sh - Sop


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
                element_labels = [f"E{e['id']}: {e['start']} -> {e['end']}" for e in st.session_state.elements]
                label = st.selectbox("Удалить стержень", options=element_labels, key="element_to_delete")
                if st.button("Удалить выбранный стержень"):
                    element_id = int(label.split(":")[0][1:])
                    ok, msg = delete_element(element_id)
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
                        element_opts = [f"E{e['id']}: {e['start']} -> {e['end']}" for e in st.session_state.elements]
                        picked = st.selectbox("Стержень", options=element_opts, key="q_element")
                        q = st.number_input("q (отрицательное = вниз)", value=-5.0, step=1.0, key="q_value")
                        if st.form_submit_button("Добавить нагрузку"):
                            element_id = int(picked.split(":")[0][1:])
                            ok, msg = add_load({"type": "distributed", "element_id": element_id, "q": float(q)})
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
                load_labels = []
                for load in st.session_state.loads:
                    if load["type"] == "point":
                        load_labels.append(f"L{load['id']}: Point @ {load['node']} (Fx={load['Fx']}, Fy={load['Fy']})")
                    elif load["type"] == "distributed":
                        load_labels.append(f"L{load['id']}: Dist @ E{load['element_id']} (q={load['q']})")
                    else:
                        load_labels.append(f"L{load['id']}: Moment @ {load['node']} (M={load['M']})")

                label = st.selectbox("Удалить нагрузку", options=load_labels, key="load_to_delete")
                if st.button("Удалить выбранную нагрузку"):
                    load_id = int(label.split(":")[0][1:])
                    ok, msg = delete_load(load_id)
                    (st.success if ok else st.error)(msg)
                st.dataframe(st.session_state.loads, use_container_width=True, hide_index=True)
            else:
                st.info("Нагрузки пока не добавлены.")

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

    if not calculate_pressed:
        return

    # Solve and show diagrams
    st.markdown("### Результаты расчета")
    if not st.session_state.elements:
        st.error("Расчет невозможен: нет стержней.")
        return
    if not st.session_state.supports:
        st.error("Расчет невозможен: не задано ни одной опоры.")
        return

    try:
        solved_ss, solve_warnings, _ = build_system_from_state()
        solved_ss.solve()
    except Exception as exc:
        st.error(f"Ошибка расчета: {exc}")
        return

    if solve_warnings:
        st.warning("Некоторые объекты были пропущены при расчете:\n- " + "\n- ".join(solve_warnings))

    reaction_rows, reaction_missed = collect_support_reaction_rows(solved_ss)
    render_static_determinacy_block()
    result_card(solved_ss, reaction_rows)
    if reaction_missed:
        st.warning("Часть реакций не удалось сопоставить:\n- " + "\n- ".join(reaction_missed))
    tab_m, tab_q, tab_n, tab_r = st.tabs(
        ["Эпюра моментов M", "Эпюра поперечных сил Q", "Эпюра продольных сил N", "Реакции опор"]
    )

    with tab_m:
        fig_m = solved_ss.show_bending_moment(show=False, verbosity=1, offset=(0.0, 0.06))
        enhance_anastruct_figure(fig_m, arrow_color="#D32F2F", arrow_linewidth=2.4)
        st.pyplot(fig_m, clear_figure=True)
    with tab_q:
        fig_q = solved_ss.show_shear_force(show=False, verbosity=1, offset=(0.0, 0.06))
        enhance_anastruct_figure(fig_q, arrow_color="#D32F2F", arrow_linewidth=2.4)
        st.pyplot(fig_q, clear_figure=True)
    with tab_n:
        fig_n = solved_ss.show_axial_force(show=False, verbosity=1, offset=(0.0, 0.06))
        enhance_anastruct_figure(fig_n, arrow_color="#D32F2F", arrow_linewidth=2.4)
        st.pyplot(fig_n, clear_figure=True)
    with tab_r:
        render_support_reactions(solved_ss)


if __name__ == "__main__":
    main()
