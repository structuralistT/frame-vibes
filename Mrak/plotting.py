"""Matplotlib figures, anaStruct figure styling, live preview drawing."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from anastruct import SystemElements
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from models import coerce_element_uuid

import state_manager as sm

# Размер фигур для эпюр M/Q/N (передаётся в anaStruct; дублирует rcParams для надёжности).
EPURE_FIGSIZE: tuple[float, float] = (12.0, 8.0)


def epure_diagram_kwargs(scale: float, *, values_deflection: float = 1.0) -> dict[str, object]:
    """
    Параметры для ``show_bending_moment`` / ``show_shear_force`` / ``show_axial_force``.

    В anaStruct **verbosity=0** включает числовые подписи на эпюре; ``values_deflection``
    увеличивает вертикальный отвод подписей (меньше наложений).
    """
    off_y = 0.06 + 0.08 * float(values_deflection)
    return {
        "show": False,
        "verbosity": 0,
        "scale": float(scale),
        "offset": (0.0, off_y),
        "figsize": EPURE_FIGSIZE,
    }


def _member_style(el) -> tuple[str, float]:
    """Visual style for regular member vs tie-rod."""
    if bool(getattr(el, "is_tie", False)):
        return "#4EA3FF", 1.6
    return "#FFFFFF", 2.5


def overlay_tie_members(fig) -> None:
    """Overlay tie-rods so they are visually distinct on anaStruct plots."""
    if not fig or not fig.axes:
        return
    ax = fig.axes[0]
    lookup = sm._node_lookup()
    for el in st.session_state.elements:
        if not bool(getattr(el, "is_tie", False)):
            continue
        na, nb = sm.element_endpoint_names(el)
        a, b = lookup.get(na), lookup.get(nb)
        if not a or not b:
            continue
        ax.plot([a.x, b.x], [a.y, b.y], color="#4EA3FF", lw=1.6, zorder=26, solid_capstyle="round")


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
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["figure.dpi"] = 150


def enhance_anastruct_figure(
    fig,
    *,
    arrow_color: str = "#D32F2F",
    arrow_linewidth: float = 2.4,
    text_offset_step: float = 0.015,
    annotation_fontsize: float | None = None,
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
    label_fs = float(annotation_fontsize) if annotation_fontsize is not None else 14.0
    for txt in ax.texts:
        txt.set_fontsize(label_fs)
        txt.set_color("#FAFAFA")
        txt.set_bbox(dict(boxstyle="round,pad=0.18", facecolor="#1f242d", edgecolor="none", alpha=0.55))
        pos = txt.get_position()
        key = (round(float(pos[0]), 6), round(float(pos[1]), 6))
        idx = seen_positions.get(key, 0)
        if idx > 0:
            txt.set_position((pos[0], pos[1] + idx * text_offset_step * y_span))
        seen_positions[key] = idx + 1

def overlay_all_session_nodes(fig, *, color: str = "#F39C12") -> None:
    """Draw every user node on top of anaStruct figure (names always visible)."""
    if not fig.axes or not st.session_state.nodes:
        return
    ax = fig.axes[0]
    for n in st.session_state.nodes:
        ax.plot(
            n.x,
            n.y,
            "o",
            ms=10,
            color=color,
            zorder=22,
            markeredgecolor="white",
            markeredgewidth=2.0,
        )
        ax.annotate(
            n.name,
            (n.x, n.y),
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
    node = sm.get_node_by_name(node_name)
    if not node:
        return

    ax = fig.axes[0]
    x, y = float(node.x), float(node.y)
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


def overlay_session_hinges(fig) -> None:
    """Draw hinge markers from session state on top of preview figure."""
    if not fig or not fig.axes or not st.session_state.hinges:
        return
    ax = fig.axes[0]
    node_lookup = sm._node_lookup()
    span = max(abs(ax.get_xlim()[1] - ax.get_xlim()[0]), abs(ax.get_ylim()[1] - ax.get_ylim()[0]), 1.0)
    # Радиус порядка видимого размера узла (ms≈10) на типичном масштабе осей.
    r = 0.007 * span

    for hinge in st.session_state.hinges:
        node_name = sm._hinge_node_name(hinge)
        node = node_lookup.get(node_name) if node_name else None
        if not node:
            continue
        circ = plt.Circle(
            (float(node.x), float(node.y)),
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
    lookup = sm._node_lookup()
    xmin, xmax, ymin, ymax = sm._preview_axis_limits()
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
        na, nb = sm.element_endpoint_names(el)
        a, b = lookup.get(na), lookup.get(nb)
        if not a or not b:
            continue
        member_color, member_lw = _member_style(el)
        ax.plot([a.x, b.x], [a.y, b.y], color=member_color, lw=member_lw, zorder=1, solid_capstyle="round")

    # Supports (simple symbols at node)
    for sup in st.session_state.supports:
        nd = lookup.get(sup["node"])
        if not nd:
            continue
        x, y = nd.x, nd.y
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
                xytext=(nd.x, nd.y),
                xy=(nd.x + fx * scale * 0.15, nd.y + fy * scale * 0.15),
                arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=2),
                zorder=8,
            )
        elif load["type"] == "moment":
            nd = lookup.get(load["node"])
            if not nd:
                continue
            m = float(load.get("M", 0))
            ax.annotate(f"M={m:g}", (nd.x, nd.y), xytext=(0, -22), textcoords="offset points", ha="center", color="#9B59B6", fontsize=10, fontweight="bold", zorder=8)
        elif load["type"] == "distributed":
            leu = coerce_element_uuid(load.get("element_id"))
            el = sm.element_by_uuid(leu) if leu is not None else None
            if not el:
                continue
            na, nb = sm.element_endpoint_names(el)
            a, b = lookup.get(na), lookup.get(nb)
            if not a or not b:
                continue
            q = float(load.get("q", 0))
            mx, my = (a.x + b.x) / 2, (a.y + b.y) / 2
            dx, dy = b.x - a.x, b.y - a.y
            ln = (dx * dx + dy * dy) ** 0.5 or 1.0
            px, py = -dy / ln, dx / ln
            off = span * 0.04 * (1 if q <= 0 else -1)
            ax.annotate(f"q={q:g}", (mx + px * off, my + py * off), ha="center", fontsize=9, color="#3498DB", zorder=8)

    # Nodes on top (above members, supports, load arrows — zorder >= 20)
    for n in nodes:
        ax.plot(
            n.x,
            n.y,
            "o",
            ms=10,
            color="#F39C12",
            zorder=22,
            markeredgecolor="white",
            markeredgewidth=2.0,
        )
        ax.annotate(
            n.name,
            (n.x, n.y),
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
        node_name = sm._hinge_node_name(hinge)
        nd = lookup.get(node_name) if node_name else None
        if not nd:
            continue
        circ = plt.Circle(
            (float(nd.x), float(nd.y)),
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
                pxmin, pxmax, pymin, pymax = sm._preview_axis_limits()
                cxmin, cxmax = ax_pv.get_xlim()
                cymin, cymax = ax_pv.get_ylim()
                ax_pv.set_xlim(min(cxmin, pxmin), max(cxmax, pxmax))
                ax_pv.set_ylim(min(cymin, pymin), max(cymax, pymax))
            overlay_all_session_nodes(fig)
            overlay_tie_members(fig)
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
