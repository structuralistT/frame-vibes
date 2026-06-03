"""Session state and model CRUD for the frame editor."""
from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import streamlit as st

import engine
from models import MemberElement, Node, coerce_element_uuid, new_member, new_node


def migrate_legacy_graph_if_needed() -> None:
    """Convert legacy dict nodes/elements to dataclasses with UUIDs; preserve hinge/load links."""
    nodes = st.session_state.get("nodes") or []
    if not nodes or isinstance(nodes[0], Node):
        return

    name_to_uuid: dict[str, UUID] = {}
    new_nodes: list[Node] = []
    for nd in nodes:
        uid = uuid4()
        name_to_uuid[str(nd["name"])] = uid
        new_nodes.append(Node(id=uid, name=str(nd["name"]), x=float(nd["x"]), y=float(nd["y"])))
    st.session_state.nodes = new_nodes

    old_el_to_uuid: dict[int, UUID] = {}
    new_elements: list[MemberElement] = []
    for e in st.session_state.elements:
        su = name_to_uuid.get(str(e["start"]))
        eu = name_to_uuid.get(str(e["end"]))
        if su is None or eu is None:
            continue
        mid = uuid4()
        old_el_to_uuid[int(e["id"])] = mid
        new_elements.append(MemberElement(id=mid, start_node_id=su, end_node_id=eu, is_tie=False))
    st.session_state.elements = new_elements

    for h in st.session_state.hinges:
        oid = h.get("element_id")
        if isinstance(oid, int) and oid in old_el_to_uuid:
            h["element_id"] = old_el_to_uuid[oid]

    for load in st.session_state.loads:
        if load.get("type") == "distributed":
            oid = load.get("element_id")
            if isinstance(oid, int) and oid in old_el_to_uuid:
                load["element_id"] = old_el_to_uuid[oid]

    if "next_element_id" in st.session_state:
        del st.session_state["next_element_id"]


def init_state() -> None:
    """Initialize all session state containers used by the editor."""
    defaults = {
        "nodes": [],
        "elements": [],
        "supports": [],
        "loads": [],
        "hinges": [],
        "next_support_id": 1,
        "next_load_id": 1,
        "next_hinge_id": 1,
        "solved_ss": None,
        "solve_warnings": [],
        "unit_result": None,
        "support_displacements": {},
        "support_settlement_rows": None,
        "displacement_tab_params": {"thermal_tau": 0.0, "thermal_tau_label": "τ"},
        "global_EI": 5000.0,
        "global_EA": 1.0e9,
        "node_name_widget_key": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    migrate_legacy_graph_if_needed()


def nodes_by_uuid() -> dict[UUID, Node]:
    return {n.id: n for n in st.session_state.nodes}


def element_endpoint_names(el: MemberElement) -> tuple[str, str]:
    by = nodes_by_uuid()
    n1 = by.get(el.start_node_id)
    n2 = by.get(el.end_node_id)
    if not n1 or not n2:
        return ("", "")
    return (n1.name, n2.name)


def element_by_uuid(raw: Any) -> MemberElement | None:
    uid = coerce_element_uuid(raw)
    if uid is None:
        return None
    return next((e for e in st.session_state.elements if e.id == uid), None)


def nodes_as_table_rows() -> list[dict[str, Any]]:
    return [{"name": n.name, "x": n.x, "y": n.y} for n in st.session_state.nodes]


def elements_as_table_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for e in st.session_state.elements:
        a, b = element_endpoint_names(e)
        rows.append({"start": a, "end": b, "tie": "Да" if bool(getattr(e, "is_tie", False)) else "Нет"})
    return rows


def get_node_by_name(name: str) -> Node | None:
    for node in st.session_state.nodes:
        if node.name == name:
            return node
    return None


def _node_name_to_sequence_index(name: str) -> int | None:
    """Map auto-names A..Z, A1..Z1, A2.. to a linear index; other names -> None."""
    s = name.strip()
    if len(s) == 1 and "A" <= s <= "Z":
        return ord(s) - ord("A")
    if len(s) >= 2 and "A" <= s[0] <= "Z" and s[1:].isdigit():
        num = int(s[1:])
        if num < 1:
            return None
        return 26 * num + (ord(s[0]) - ord("A"))
    return None


def _sequence_index_to_node_name(idx: int) -> str:
    """Inverse of ``_node_name_to_sequence_index`` (idx >= 0)."""
    if idx < 26:
        return chr(ord("A") + idx)
    tier = idx // 26
    rem = idx % 26
    return f"{chr(ord('A') + rem)}{tier}"


def get_next_node_label() -> str:
    """
    Следующая метка узла: A, B, …, Z, затем A1, B1, …, Z1, A2, …

    Берётся максимальный индекс среди имён, попадающих в эту схему; остальные имена игнорируются.
    """
    nodes = st.session_state.get("nodes") or []
    indices: list[int] = []
    for n in nodes:
        if isinstance(n, Node):
            idx = _node_name_to_sequence_index(n.name)
            if idx is not None:
                indices.append(idx)
    if not indices:
        return "A"
    return _sequence_index_to_node_name(max(indices) + 1)


def get_next_node_name() -> str:
    """Обратная совместимость: то же, что ``get_next_node_label``."""
    return get_next_node_label()


def add_node(name: str, x: float, y: float) -> tuple[bool, str]:
    clean = name.strip()
    if not clean:
        return False, "Имя узла не может быть пустым."
    if get_node_by_name(clean):
        return False, f"Узел '{clean}' уже существует."
    st.session_state.nodes.append(new_node(clean, x, y))
    return True, f"Узел '{clean}' добавлен."


def delete_node(node_name: str) -> tuple[bool, str]:
    target = get_node_by_name(node_name)
    if not target:
        return False, "Узел не найден."

    nid = target.id
    st.session_state.nodes = [n for n in st.session_state.nodes if n.id != nid]
    removed_element_ids = {
        e.id for e in st.session_state.elements if e.start_node_id == nid or e.end_node_id == nid
    }
    st.session_state.elements = [
        e for e in st.session_state.elements if e.start_node_id != nid and e.end_node_id != nid
    ]
    st.session_state.hinges = [
        h for h in st.session_state.hinges if coerce_element_uuid(h.get("element_id")) not in removed_element_ids
    ]
    st.session_state.supports = [s for s in st.session_state.supports if s["node"] != node_name]
    st.session_state.loads = [
        load
        for load in st.session_state.loads
        if not (
            (load["type"] in ("point", "moment") and load.get("node") == node_name)
            or (load["type"] == "distributed" and coerce_element_uuid(load.get("element_id")) in removed_element_ids)
        )
    ]
    return True, f"Узел '{node_name}' и связанные объекты удалены."


def add_element(start_node: str, end_node: str) -> tuple[bool, str]:
    if start_node == end_node:
        return False, "Начальный и конечный узлы должны отличаться."

    n1 = get_node_by_name(start_node)
    n2 = get_node_by_name(end_node)
    if not n1 or not n2:
        return False, "Узел не найден."

    for element in st.session_state.elements:
        same_direct = element.start_node_id == n1.id and element.end_node_id == n2.id
        same_reverse = element.start_node_id == n2.id and element.end_node_id == n1.id
        if same_direct or same_reverse:
            return False, "Такой стержень уже существует."

    new_el = new_member(n1.id, n2.id)
    st.session_state.elements.append(new_el)
    num = _element_number_by_id().get(new_el.id, 1)
    return True, f"Стержень {num} добавлен."


def delete_element(element_id: UUID) -> tuple[bool, str]:
    if not any(e.id == element_id for e in st.session_state.elements):
        return False, "Стержень не найден."

    num = _element_number_by_id().get(element_id, 0)
    st.session_state.elements = [e for e in st.session_state.elements if e.id != element_id]
    st.session_state.hinges = [
        h for h in st.session_state.hinges if coerce_element_uuid(h.get("element_id")) != element_id
    ]
    st.session_state.loads = [
        load
        for load in st.session_state.loads
        if not (load["type"] == "distributed" and coerce_element_uuid(load.get("element_id")) == element_id)
    ]
    return True, f"Стержень {num or '?'} и связанные нагрузки удалены."


def set_element_tie(element_id: UUID, is_tie: bool) -> tuple[bool, str]:
    """Toggle tie-rod flag for an element and enforce hinges for tie-rods."""
    target_idx = next((i for i, e in enumerate(st.session_state.elements) if e.id == element_id), None)
    if target_idx is None:
        return False, "Стержень не найден."

    el = st.session_state.elements[target_idx]
    st.session_state.elements[target_idx] = replace(el, is_tie=bool(is_tie))
    el_num = _element_number_by_id().get(element_id, 0)

    if is_tie:
        add_or_replace_hinge(element_id, "start")
        add_or_replace_hinge(element_id, "end")
        return True, f"Стержень {el_num or '?'} помечен как затяжка (добавлены шарниры с обоих концов)."

    return True, f"Стержень {el_num or '?'} больше не помечен как затяжка."


def add_support(node_name: str, support_type: str, angle_deg: float) -> tuple[bool, str]:
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
    if not any(int(l["id"]) == int(load_id) for l in st.session_state.loads):
        return False, "Нагрузка не найдена."
    st.session_state.loads = [l for l in st.session_state.loads if int(l["id"]) != int(load_id)]
    return True, f"Нагрузка L{load_id} удалена."


def _hinge_node_name(hinge: dict) -> str | None:
    """Return node name where hinge is attached (based on element end)."""
    el = element_by_uuid(hinge.get("element_id"))
    if not el:
        return None
    a, b = element_endpoint_names(el)
    if hinge.get("position") == "start":
        return a
    if hinge.get("position") == "end":
        return b
    return None


def add_or_replace_hinge(element_id: UUID, position: str) -> tuple[bool, str]:
    if position not in ("start", "end"):
        return False, "Некорректное положение шарнира."

    element = next((e for e in st.session_state.elements if e.id == element_id), None)
    if not element:
        return False, "Выбранный стержень не найден."

    st.session_state.hinges = [
        h
        for h in st.session_state.hinges
        if not (coerce_element_uuid(h.get("element_id")) == element_id and h.get("position") == position)
    ]

    hinge_id = st.session_state.next_hinge_id
    st.session_state.next_hinge_id += 1
    st.session_state.hinges.append({"id": hinge_id, "element_id": element_id, "position": position})

    num_map = _element_number_by_id()
    el_num = num_map.get(element.id, 0)
    pos_ru = "начало" if position == "start" else "конец"
    return True, f"Шарнир добавлен на стержень {el_num} ({pos_ru})."


def add_hinges_at_node(node_name: str) -> tuple[bool, str]:
    target = get_node_by_name(node_name)
    if not target:
        return False, "Узел не найден."
    nid = target.id
    incident: list[tuple[UUID, str]] = []
    for e in st.session_state.elements:
        if e.start_node_id == nid:
            incident.append((e.id, "start"))
        elif e.end_node_id == nid:
            incident.append((e.id, "end"))
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


def _preview_axis_limits() -> tuple[float, float, float, float]:
    """Padding around all node coordinates for consistent axes."""
    nodes = st.session_state.nodes
    if not nodes:
        return -1.0, 1.0, -1.0, 1.0
    xs = [float(n.x) for n in nodes]
    ys = [float(n.y) for n in nodes]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    pad = 0.18 * span
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def _node_lookup() -> dict[str, Node]:
    return {n.name: n for n in st.session_state.nodes}


def _fmt_num(v: float) -> str:
    return f"{float(v):.1f}"


def _force_arrow(fx: float, fy: float) -> str:
    if abs(fx) >= abs(fy) and abs(fx) > 1e-12:
        return "→" if fx > 0 else "←"
    if abs(fy) > 1e-12:
        return "↑" if fy > 0 else "↓"
    return "•"


def _element_label_for_load(element_id: UUID) -> str:
    el = element_by_uuid(element_id)
    if not el:
        return str(element_id)[:8]
    a, b = element_endpoint_names(el)
    return f"{a}{b}"


def _element_number_by_id() -> dict[UUID, int]:
    return {e.id: i for i, e in enumerate(st.session_state.elements, start=1)}


def _element_ui_label(element: MemberElement) -> str:
    num_map = _element_number_by_id()
    num = num_map.get(element.id, 0)
    a, b = element_endpoint_names(element)
    return f"Стержень {num} ({a} → {b})"


def _hinge_ui_label(hinge: dict) -> str:
    element = element_by_uuid(hinge.get("element_id"))
    if not element:
        el_text = f"Стержень {hinge.get('element_id')}"
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
        eu = coerce_element_uuid(load.get("element_id"))
        el_num = num_map.get(eu, 0) if eu else 0
        q = float(load.get("q", 0.0))
        q_arrow = "↓" if q < 0 else "↑"
        return f"q_{el_num} = {_fmt_num(abs(q))} кН/м ({q_arrow})"

    node = load["node"]
    m = float(load.get("M", 0.0))
    return f"M_{node} = {_fmt_num(abs(m))} кН·м (↻)"


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


def _dict_has_nonzero_settlements(sd: dict[Any, Any]) -> bool:
    eps_mm = 1e-9
    eps_rad = 1e-12
    for v in sd.values():
        if not isinstance(v, dict):
            continue
        if abs(float(v.get("dn_mm", 0.0))) > eps_mm:
            return True
        if abs(float(v.get("dx_mm", 0.0))) > eps_mm:
            return True
        if abs(float(v.get("dy_mm", 0.0))) > eps_mm:
            return True
        if abs(float(v.get("phi_rad", 0.0))) > eps_rad:
            return True
    return False


def _default_settlement_rows_for_supports(supports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for support in supports:
        sid = int(support["id"])
        node = str(support["node"])
        s_type = str(support["type"])
        if s_type == "roller":
            rows.append(
                {
                    "support_id": sid,
                    "node_name": node,
                    "support_type": s_type,
                    "direction": "N",
                    "symbol": f"c_{node}",
                    "value": 0.0,
                }
            )
        elif s_type == "hinged":
            rows.extend(
                [
                    {
                        "support_id": sid,
                        "node_name": node,
                        "support_type": s_type,
                        "direction": "X",
                        "symbol": f"c_{node}x",
                        "value": 0.0,
                    },
                    {
                        "support_id": sid,
                        "node_name": node,
                        "support_type": s_type,
                        "direction": "Y",
                        "symbol": f"c_{node}y",
                        "value": 0.0,
                    },
                ]
            )
        else:
            rows.extend(
                [
                    {
                        "support_id": sid,
                        "node_name": node,
                        "support_type": s_type,
                        "direction": "X",
                        "symbol": f"c_{node}x",
                        "value": 0.0,
                    },
                    {
                        "support_id": sid,
                        "node_name": node,
                        "support_type": s_type,
                        "direction": "Y",
                        "symbol": f"c_{node}y",
                        "value": 0.0,
                    },
                    {
                        "support_id": sid,
                        "node_name": node,
                        "support_type": s_type,
                        "direction": "Rot",
                        "symbol": f"φ_{node}",
                        "value": 0.0,
                    },
                ]
            )
    return rows


def _settlement_row_key(row: dict[str, Any]) -> tuple[int, str]:
    return (int(row["support_id"]), str(row["direction"]))


def _merge_settlement_rows_with_supports(
    raw_existing: list[dict[str, Any]] | None,
    supports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    defaults = _default_settlement_rows_for_supports(supports)
    if not raw_existing:
        return defaults
    prev = {_settlement_row_key(r): r for r in raw_existing if "support_id" in r and "direction" in r}
    merged: list[dict[str, Any]] = []
    for d in defaults:
        key = _settlement_row_key(d)
        if key in prev:
            old = prev[key]
            sym = str(old.get("symbol", d["symbol"])).strip() or str(d["symbol"])
            try:
                val = float(old.get("value", 0.0))
            except (TypeError, ValueError):
                val = 0.0
            merged.append({**d, "symbol": sym, "value": val})
        else:
            merged.append(dict(d))
    return merged


def rows_from_legacy_displacement_dict(
    supports: list[dict[str, Any]],
    sd: dict[Any, Any],
) -> list[dict[str, Any]]:
    """Одноразовая миграция старого ``support_displacements`` в табличные строки."""
    rows: list[dict[str, Any]] = []
    for support in supports:
        sid = int(support["id"])
        node = str(support["node"])
        s_type = str(support["type"])
        raw = sd.get(sid, {})
        disp = _migrate_support_displacement_entry(support, raw if isinstance(raw, dict) else {})
        if s_type == "roller":
            rows.append(
                {
                    "support_id": sid,
                    "node_name": node,
                    "support_type": s_type,
                    "direction": "N",
                    "symbol": str(disp.get("sym_dn", f"c_{node}")).strip() or f"c_{node}",
                    "value": float(disp.get("dn_mm", 0.0)) / 1000.0,
                }
            )
        elif s_type == "hinged":
            rows.extend(
                [
                    {
                        "support_id": sid,
                        "node_name": node,
                        "support_type": s_type,
                        "direction": "X",
                        "symbol": str(disp.get("sym_dx", f"c_{node}x")).strip() or f"c_{node}x",
                        "value": float(disp.get("dx_mm", 0.0)) / 1000.0,
                    },
                    {
                        "support_id": sid,
                        "node_name": node,
                        "support_type": s_type,
                        "direction": "Y",
                        "symbol": str(disp.get("sym_dy", f"c_{node}y")).strip() or f"c_{node}y",
                        "value": float(disp.get("dy_mm", 0.0)) / 1000.0,
                    },
                ]
            )
        else:
            rows.extend(
                [
                    {
                        "support_id": sid,
                        "node_name": node,
                        "support_type": s_type,
                        "direction": "X",
                        "symbol": str(disp.get("sym_dx", f"c_{node}x")).strip() or f"c_{node}x",
                        "value": float(disp.get("dx_mm", 0.0)) / 1000.0,
                    },
                    {
                        "support_id": sid,
                        "node_name": node,
                        "support_type": s_type,
                        "direction": "Y",
                        "symbol": str(disp.get("sym_dy", f"c_{node}y")).strip() or f"c_{node}y",
                        "value": float(disp.get("dy_mm", 0.0)) / 1000.0,
                    },
                    {
                        "support_id": sid,
                        "node_name": node,
                        "support_type": s_type,
                        "direction": "Rot",
                        "symbol": str(disp.get("sym_phi", f"φ_{node}")).strip() or f"φ_{node}",
                        "value": float(disp.get("phi_rad", 0.0)),
                    },
                ]
            )
    return rows


def ensure_support_displacements_state() -> None:
    """Синхронизирует таблицу осадок с опорами и обновляет ``support_displacements`` для расчётов."""
    supports = st.session_state.supports
    sd = st.session_state.support_displacements
    if not isinstance(sd, dict):
        st.session_state.support_displacements = {}
        sd = st.session_state.support_displacements

    active_ids = {int(s["id"]) for s in supports}
    for sid in list(sd.keys()):
        if int(sid) not in active_ids:
            sd.pop(sid, None)

    raw_rows = st.session_state.get("support_settlement_rows")
    if raw_rows is None:
        if _dict_has_nonzero_settlements(sd):
            merged = rows_from_legacy_displacement_dict(supports, sd)
        else:
            merged = _default_settlement_rows_for_supports(supports)
        st.session_state.support_settlement_rows = merged
    else:
        st.session_state.support_settlement_rows = _merge_settlement_rows_with_supports(
            raw_rows if isinstance(raw_rows, list) else None,
            supports,
        )
        merged = st.session_state.support_settlement_rows

    st.session_state.support_displacements = engine.support_settlement_rows_to_map(merged, supports)


def commit_support_settlement_from_dataframe(df: Any) -> None:
    """Записывает результат ``st.data_editor`` в ``support_settlement_rows`` и карту осадок."""
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        raw_val = row.get("value", 0.0)
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            val = 0.0
        if val != val:  # NaN
            val = 0.0
        sym = row.get("symbol", "")
        if sym is None or (isinstance(sym, float) and sym != sym):
            sym = ""
        rows.append(
            {
                "support_id": int(row["support_id"]),
                "node_name": str(row["node_name"]),
                "support_type": str(row["support_type"]),
                "direction": str(row["direction"]),
                "symbol": str(sym).strip(),
                "value": val,
            }
        )
    st.session_state.support_settlement_rows = rows
    st.session_state.support_displacements = engine.support_settlement_rows_to_map(rows, st.session_state.supports)
