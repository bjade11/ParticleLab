"""분석 파라미터 dataclass (DESIGN.md §1 과 1:1 대응).

모든 dataclass 는 ``from_dict`` / ``to_dict`` 를 제공하며, 입력 dict 에 없는 키는 기본값을 쓰고
알 수 없는 키는 무시한다 (프런트/백엔드 스키마가 조금 앞서가도 깨지지 않도록).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping, Optional, Tuple, TypeVar

T = TypeVar("T", bound="_Base")


class _Base:
    """from_dict / to_dict 공용 믹스인."""

    @classmethod
    def from_dict(cls: type[T], d: Optional[Mapping[str, Any]]) -> T:
        d = d or {}
        kwargs: dict[str, Any] = {}
        for f in fields(cls):  # type: ignore[arg-type]
            if f.name in d:
                kwargs[f.name] = d[f.name]
        return cls(**kwargs)  # type: ignore[call-arg]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScaleParams(_Base):
    mode: str = "auto"              # "auto" | "manual"
    pixel_to_micro: float = 0.198   # manual 일 때 μm/px (auto 실패 fallback 으로도 사용)
    auto_default: float = 0.7       # auto 모드에서 파일명 매칭 실패 시 값


@dataclass
class SemParams(_Base):
    bottom_crop_ratio: float = 0.074
    brightness: int = 0
    contrast: float = 1.0
    gamma: float = 1.0
    use_clahe: bool = False
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    use_usm: bool = False
    usm_amount: float = 0.6
    usm_radius: float = 1.5


@dataclass
class EdgeParams(_Base):
    weight: float = 0.5
    kernel: int = 3
    method: str = "laplacian"       # laplacian | sobelx | sobely | scharr


@dataclass
class MorphOp(_Base):
    kernel: int = 3
    iterations: int = 1


@dataclass
class MorphologyParams(_Base):
    open: MorphOp = field(default_factory=MorphOp)
    close: MorphOp = field(default_factory=MorphOp)
    erode: MorphOp = field(default_factory=MorphOp)
    dilate: MorphOp = field(default_factory=MorphOp)

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "MorphologyParams":
        d = d or {}
        return cls(
            open=_as(MorphOp, d.get("open")),
            close=_as(MorphOp, d.get("close")),
            erode=_as(MorphOp, d.get("erode")),
            dilate=_as(MorphOp, d.get("dilate")),
        )


@dataclass
class MarkerParams(_Base):
    peak_min_distance: int = 5
    distance_threshold: float = 0.3
    dilate_kernel: int = 3
    dilate_iter: int = 1
    erosion_kernel: int = 5
    erosion_iter: int = 2
    blob_min_sigma: float = 3.0
    blob_max_sigma: float = 30.0
    blob_num_sigma: int = 10
    blob_threshold: float = 0.1


@dataclass
class WatershedParams(_Base):
    blur_kernel: Tuple[int, int] = (3, 3)
    enhance_edges: bool = False
    edge: EdgeParams = field(default_factory=EdgeParams)
    morphology: Optional[MorphologyParams] = field(default_factory=MorphologyParams)  # None 이면 생략
    marker_method: str = "peak_local_max"
    marker: MarkerParams = field(default_factory=MarkerParams)
    min_size: int = 30

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "WatershedParams":
        d = d or {}
        bk = d.get("blur_kernel", (3, 3))
        if isinstance(bk, (int, float)):
            bk = (int(bk), int(bk))
        bk = (int(bk[0]), int(bk[1]))
        morph_raw = d.get("morphology", {})
        morphology = None if morph_raw is None else MorphologyParams.from_dict(morph_raw)
        return cls(
            blur_kernel=bk,
            enhance_edges=bool(d.get("enhance_edges", False)),
            edge=_as(EdgeParams, d.get("edge")),
            morphology=morphology,
            marker_method=str(d.get("marker_method", "peak_local_max")),
            marker=_as(MarkerParams, d.get("marker")),
            min_size=int(d.get("min_size", 30)),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["blur_kernel"] = list(self.blur_kernel)
        return d


@dataclass
class HybridEngineParams(_Base):
    high_thresh: float = 0.75
    low_thresh: float = 0.40
    boundary_thickness: int = 3
    min_anchor_area: int = 10
    balance_floor: float = 0.25
    tau_scale: float = 1.0
    yolo_round_strength: int = 3
    yolo_remove_overlaps: bool = True
    yolo_remove_enclosed: bool = True
    enclosed_ratio: float = 0.9
    yolo_conf: float = 0.15
    yolo_iou: float = 0.5
    yolo_imgsz: int = 640
    unet_imgsz: Optional[int] = None
    device: str = "auto"
    unet_weights: str = "basic"
    yolo_weights: str = "basic"


@dataclass
class MeasureParams(_Base):
    filter_border_particles: bool = True
    min_contour_points: int = 30
    min_diameter: float = 0.0
    exclude_linear_contour: bool = False
    linear_contour_percentile: float = 10.0
    exclude_dark_particle: bool = False
    dark_particle_percentile: float = 10.0
    curvature_percentile: float = 0.0   # 0 이면 끔


@dataclass
class AfterFilterParams(_Base):
    diameter_lower_percentile: float = 0.0
    diameter_upper_percentile: float = 0.0
    enable_circularity: bool = True     # False 면 원형도 필터 끔
    circularity_mode: str = "value"     # value | percentile
    circularity_value: float = 0.7
    circularity_percentile: float = 5.0
    enable_aspect_ratio: bool = True    # False 면 종횡비 필터 끔
    aspect_ratio_mode: str = "value"
    aspect_ratio_value: float = 2.0
    aspect_ratio_percentile: float = 5.0


@dataclass
class SpatialParams(_Base):
    enable_boundary: bool = False
    boundary_contact_threshold: float = 2.0
    enable_nnd: bool = False
    enable_dbscan: bool = False
    dbscan_eps_auto: bool = True
    dbscan_eps_factor: float = 2.5
    dbscan_eps_value: float = 5.0
    dbscan_min_samples_auto: bool = True
    dbscan_min_samples_factor: float = 0.01
    dbscan_min_samples_value: int = 2


@dataclass
class OutputParams(_Base):
    make_ppt: bool = True
    make_contour_only: bool = True
    make_normal_dist: bool = True
    jpeg_quality: int = 95


@dataclass
class AnalysisParams(_Base):
    engine: str = "watershed"           # "watershed" | "ai_hybrid"
    scale: ScaleParams = field(default_factory=ScaleParams)
    sem: SemParams = field(default_factory=SemParams)
    watershed: WatershedParams = field(default_factory=WatershedParams)
    hybrid: HybridEngineParams = field(default_factory=HybridEngineParams)
    measure: MeasureParams = field(default_factory=MeasureParams)
    after_filter: AfterFilterParams = field(default_factory=AfterFilterParams)
    spatial: SpatialParams = field(default_factory=SpatialParams)
    output: OutputParams = field(default_factory=OutputParams)

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "AnalysisParams":
        d = d or {}
        return cls(
            engine=str(d.get("engine", "watershed")),
            scale=_as(ScaleParams, d.get("scale")),
            sem=_as(SemParams, d.get("sem")),
            watershed=WatershedParams.from_dict(d.get("watershed")),
            hybrid=_as(HybridEngineParams, d.get("hybrid")),
            measure=_as(MeasureParams, d.get("measure")),
            after_filter=_as(AfterFilterParams, d.get("after_filter")),
            spatial=_as(SpatialParams, d.get("spatial")),
            output=_as(OutputParams, d.get("output")),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["watershed"] = self.watershed.to_dict()
        return d


def _as(cls: type[T], v: Any) -> T:
    """dict 또는 이미 dataclass 인스턴스인 값을 cls 로 변환."""
    if isinstance(v, cls):
        return v
    return cls.from_dict(v)  # type: ignore[return-value]


__all__ = [
    "ScaleParams", "SemParams", "EdgeParams", "MorphOp", "MorphologyParams", "MarkerParams",
    "WatershedParams", "HybridEngineParams", "MeasureParams", "AfterFilterParams",
    "SpatialParams", "OutputParams", "AnalysisParams",
]
