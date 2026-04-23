"""Domain models for the frame editor (UUID identity for graph topology)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, TypedDict
from uuid import UUID, uuid4


def coerce_element_uuid(raw: Any) -> UUID | None:
    """Normalize hinge / load ``element_id`` to ``UUID``."""
    if raw is None:
        return None
    if isinstance(raw, UUID):
        return raw
    if isinstance(raw, str):
        try:
            return UUID(raw)
        except ValueError:
            return None
    return None


@dataclass(frozen=True, slots=True)
class Node:
    """Planar frame node: topology uses ``id``, labels use ``name``."""

    id: UUID
    name: str
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class MemberElement:
    """Frame member between two nodes (endpoints by node UUID, not coordinates)."""

    id: UUID
    start_node_id: UUID
    end_node_id: UUID
    is_tie: bool = False


def new_node(name: str, x: float, y: float) -> Node:
    return Node(id=uuid4(), name=name.strip(), x=float(x), y=float(y))


def new_member(start_node_id: UUID, end_node_id: UUID) -> MemberElement:
    return MemberElement(id=uuid4(), start_node_id=start_node_id, end_node_id=end_node_id, is_tie=False)


# --- Session / engine payloads (plain dicts with known shape) ---------------------------------

SupportType = Literal["fixed", "hinged", "roller"]


class SupportEntry(TypedDict, total=False):
    """Support as stored in session_state / passed to ``engine.build_system``."""

    id: int
    node: str
    type: SupportType
    angle: float


class HingeEntry(TypedDict, total=False):
    id: int
    element_id: UUID | str | int
    position: str


class PointLoadEntry(TypedDict, total=False):
    id: int
    type: Literal["point"]
    node: str
    Fx: float
    Fy: float


class MomentLoadEntry(TypedDict, total=False):
    id: int
    type: Literal["moment"]
    node: str
    M: float


class DistributedLoadEntry(TypedDict, total=False):
    id: int
    type: Literal["distributed"]
    element_id: UUID | str | int
    q: float


LoadEntry = PointLoadEntry | MomentLoadEntry | DistributedLoadEntry


class SupportDisplacementEntry(TypedDict, total=False):
    """Per-support settlement / rotation input (mm, rad, symbolic labels)."""

    sym_dn: str
    dn_mm: float
    sym_dx: str
    sym_dy: str
    sym_phi: str
    dx_mm: float
    dy_mm: float
    phi_rad: float


SupportDisplacementsMap = dict[int, SupportDisplacementEntry]


SettlementDirection = Literal["X", "Y", "Rot", "N"]


class SupportSettlementRow(TypedDict, total=False):
    """Одна строка осадки опоры для вкладки «Перемещения» и метода сил."""

    support_id: int
    node_name: str
    support_type: SupportType
    direction: SettlementDirection
    symbol: str
    value: float


class SupportCounts(TypedDict):
    """Counts of supports by kind (used in static indeterminacy estimate)."""

    fixed: int
    hinged: int
    roller: int


ReactionTableRow = TypedDict(
    "ReactionTableRow",
    {
        "Узел": str,
        "Индекс": str,
        "Тип опоры": str,
        "Тип опоры (код)": str,
        "Fx": float,
        "Fy": float,
        "Mz (Tz)": float,
        "Направление Fx": str,
        "Направление Fy": str,
        "Направление Mz": str,
    },
    total=False,
)


MohrIntegralRow = TypedDict(
    "MohrIntegralRow",
    {
        "Элемент": str,
        "L, м": float,
        "EI": float,
        "Площадь Mгр, кН·м²": float,
        "Ордината Mед (взв.)": float,
        "∫(Mгр*Mед)dx": float,
        "Вклад ∫/EI": float,
    },
    total=False,
)


class ForceMethodReport(TypedDict, total=False):
    """Return value of ``engine.build_force_method_report``."""

    ok: bool
    markdown_intro: list[str]
    latex_lines: list[str]
    markdown_solution: list[str]
    warnings: list[str]
    n_used: int
    n_target: int
    X: list[float]
    cond: float
    tie_os_used: bool
    tie_labels_ru: list[str]
    kinematic_W: int
    delta_is: list[float]
    delta_is_detail_latex: list[str]


class KinematicAnalysisResult(TypedDict):
    """Чебышёв: W = 3D − 2Ш − C₀ и расшифровка."""

    W: int
    D: int
    Sh: int
    C0: int


UnitDirection = Literal["x", "y", "rz"]


class NodeDisplacement(TypedDict, total=False):
    """Keys commonly returned by anaStruct ``get_node_displacements`` (м)."""

    ux: float
    uy: float
    phi_z: float


VereshchaginReport: TypeAlias = tuple[
    list[str],
    str,
    str,
    float,
    list[str],
    str,
    bool,
    float,
    float | None,
]
