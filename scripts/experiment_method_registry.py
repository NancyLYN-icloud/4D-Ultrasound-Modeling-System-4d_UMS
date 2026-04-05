from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class MethodProfile:
    name: str
    description: str
    cli_overrides: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodSpec:
    display_name: str
    slug: str
    paradigm: str
    academic_name: str
    paper_role: str
    implementation_method: str
    summary: str
    is_external: bool
    execution_family: str = "dynamic"
    default_profiles: dict[str, MethodProfile] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()

    def matches(self, token: str) -> bool:
        normalized = _normalize_token(token)
        if normalized in {_normalize_token(self.display_name), _normalize_token(self.slug), _normalize_token(self.academic_name)}:
            return True
        return any(normalized == _normalize_token(alias) for alias in self.aliases)

    def to_metadata(self) -> dict[str, object]:
        return {
            "display_name": self.display_name,
            "slug": self.slug,
            "paradigm": self.paradigm,
            "academic_name": self.academic_name,
            "paper_role": self.paper_role,
            "implementation_method": self.implementation_method,
            "summary": self.summary,
            "is_external": self.is_external,
            "execution_family": self.execution_family,
            "profiles": {
                name: {
                    "name": profile.name,
                    "description": profile.description,
                    "cli_overrides": profile.cli_overrides,
                }
                for name, profile in self.default_profiles.items()
            },
            "aliases": list(self.aliases),
        }


def _normalize_token(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def _profile(name: str, description: str, **kwargs: object) -> MethodProfile:
    return MethodProfile(name=name, description=description, cli_overrides=kwargs)


METHOD_SPECS: tuple[MethodSpec, ...] = (
    MethodSpec(
        display_name="动态共享-参考对应正则",
        slug="refcorr",
        paradigm="reference-correspondence regularized canonical deformation",
        academic_name="Adapted DIT Baseline",
        paper_role="external-baseline",
        implementation_method="动态共享-参考对应正则",
        summary="Representative baseline for canonical/reference-shape dynamic reconstruction with explicit correspondence regularization.",
        is_external=True,
        aliases=("reference-correspondence", "canonical-deform", "ref-corr", "external-refcorr", "dit", "dit-style", "adapted-dit"),
        default_profiles={
            "historical_best_eqbudget": _profile(
                "historical_best_eqbudget",
                "Equal-budget configuration aligned with the current best-performing ref-corr baseline.",
                canonical_hidden_dim=128,
                canonical_hidden_layers=4,
                deformation_hidden_dim=128,
                deformation_hidden_layers=3,
                confidence_floor=0.2,
                temporal_weight=0.10,
                temporal_acceleration_weight=0.05,
                phase_consistency_weight=0.05,
                correspondence_temporal_weight=0.01,
                correspondence_acceleration_weight=0.005,
                correspondence_phase_consistency_weight=0.005,
                correspondence_start_fraction=0.4,
                correspondence_ramp_fraction=0.2,
                periodicity_weight=0.10,
                deformation_weight=0.01,
                unsupported_anchor_weight=0.05,
                base_mesh_train_steps=100,
                smoothing_iterations=10,
            ),
        },
    ),
    MethodSpec(
        display_name="动态共享-连续形变场",
        slug="continuous",
        paradigm="continuous deformation-field baseline",
        academic_name="Adapted DIF Baseline",
        paper_role="external-baseline",
        implementation_method="动态共享-连续形变场",
        summary="Representative weak-structure-prior dynamic deformation-field baseline for temporally continuous reconstruction.",
        is_external=True,
        aliases=("continuous-field", "continuous-deformation-field", "external-continuous", "dif", "dif-style", "adapted-dif"),
        default_profiles={
            "historical_best_eqbudget": _profile(
                "historical_best_eqbudget",
                "Equal-budget configuration aligned with the current best-performing continuous-field baseline.",
                bootstrap_offset_weight=0.12,
                bootstrap_decay_fraction=0.30,
                unsupported_anchor_weight=0.03,
                unsupported_laplacian_weight=0.08,
                temporal_weight=0.02,
                temporal_acceleration_weight=0.01,
                phase_consistency_weight=0.01,
                correspondence_temporal_weight=0.01,
                correspondence_acceleration_weight=0.005,
                correspondence_phase_consistency_weight=0.005,
                correspondence_start_fraction=0.4,
                correspondence_ramp_fraction=0.2,
                periodicity_weight=0.02,
                deformation_weight=0.002,
                confidence_floor=1.0,
                smoothing_iterations=10,
            ),
        },
    ),
    MethodSpec(
        display_name="动态共享-解耦运动潜码",
        slug="decoupled_motion",
        paradigm="decoupled motion-shape baseline",
        academic_name="Adapted Decoupled Motion-Shape Baseline",
        paper_role="external-baseline",
        implementation_method="动态共享-解耦运动潜码",
        summary="Representative latent-variable baseline that decouples motion dynamics from shared geometry.",
        is_external=True,
        aliases=("decoupled-motion", "motion-latent", "external-decoupled-motion", "decoupled-motion-shape", "adapted-decoupled", "myocardium-baseline"),
        default_profiles={
            "historical_best_eqbudget": _profile(
                "historical_best_eqbudget",
                "Equal-budget configuration aligned with the current best-performing decoupled-motion baseline.",
                confidence_floor=1.0,
                temporal_weight=0.02,
                temporal_acceleration_weight=0.01,
                phase_consistency_weight=0.01,
                correspondence_temporal_weight=0.01,
                correspondence_acceleration_weight=0.005,
                correspondence_phase_consistency_weight=0.005,
                correspondence_start_fraction=0.4,
                correspondence_ramp_fraction=0.2,
                periodicity_weight=0.02,
                deformation_weight=0.002,
                bootstrap_offset_weight=0.25,
                bootstrap_decay_fraction=0.35,
                unsupported_anchor_weight=0.05,
                unsupported_laplacian_weight=0.12,
                motion_mean_weight=0.05,
                motion_lipschitz_weight=0.02,
                smoothing_iterations=10,
            ),
        },
    ),
    MethodSpec(
        display_name="静态增强",
        slug="phase_normalized_static",
        paradigm="phase-normalized phase-wise static reconstruction baseline",
        academic_name="Phase-Normalized Static Baseline",
        paper_role="supplementary-baseline",
        implementation_method="静态增强",
        summary="Phase-canonicalized per-phase static reconstruction baseline without shared-topology dynamic coupling.",
        is_external=False,
        execution_family="static",
        aliases=("static-enhanced", "phase-normalized-static", "supp-static", "static-phase-wise"),
    ),
    MethodSpec(
        display_name="动态共享-无先验4D场",
        slug="prior_free_4d_field",
        paradigm="prior-free spatiotemporal implicit field baseline",
        academic_name="Prior-Free 4D Field Baseline",
        paper_role="supplementary-baseline",
        implementation_method="动态共享-无先验4D场",
        summary="Observation-driven 4D implicit field baseline without shared template or structured dynamic priors.",
        is_external=True,
        aliases=("prior-free", "prior-free-4d", "supp-prior-free", "4d-field"),
    ),
    MethodSpec(
        display_name="动态共享-CPD对应点",
        slug="cpd_guided_correspondence",
        paradigm="cpd-guided correspondence baseline",
        academic_name="CPD-Guided Correspondence Baseline",
        paper_role="supplementary-baseline",
        implementation_method="动态共享-CPD对应点",
        summary="Dynamic shared-topology baseline with stronger CPD-style correspondence guidance on the shared reference surface.",
        is_external=True,
        aliases=("cpd", "cpd-guided", "cpd-correspondence", "supp-cpd"),
    ),
    MethodSpec(
        display_name="动态共享-全局基残差",
        slug="global_basis_residual",
        paradigm="shared-topology global-basis residual model",
        academic_name="Global-Basis Residual (Ours)",
        paper_role="ours",
        implementation_method="动态共享-全局基残差",
        summary="Proposed structured dynamic model combining shared topology, low-rank global motion bases, and phase-conditioned residual fields.",
        is_external=False,
        aliases=("ours", "global-basis", "gbr", "shared-topology-global-basis-residual"),
        default_profiles={
            "historical_best_eqbudget": _profile(
                "historical_best_eqbudget",
                "Equal-budget configuration aligned with the current best-performing global-basis residual model.",
                global_motion_basis_rank=6,
                confidence_floor=1.0,
                temporal_weight=0.02,
                temporal_acceleration_weight=0.01,
                phase_consistency_weight=0.01,
                correspondence_temporal_weight=0.016,
                correspondence_acceleration_weight=0.008,
                correspondence_phase_consistency_weight=0.008,
                correspondence_global_only=True,
                correspondence_bootstrap_gate=True,
                correspondence_bootstrap_gate_strength=2.0,
                correspondence_start_fraction=0.25,
                correspondence_ramp_fraction=0.15,
                periodicity_weight=0.02,
                deformation_weight=0.002,
                bootstrap_offset_weight=0.10,
                bootstrap_decay_fraction=0.25,
                basis_coefficient_bootstrap_weight=0.12,
                basis_temporal_weight=0.05,
                basis_acceleration_weight=0.025,
                basis_periodicity_weight=0.04,
                residual_mean_weight=0.02,
                residual_basis_projection_weight=0.04,
                unsupported_anchor_weight=0.02,
                unsupported_laplacian_weight=0.05,
                unsupported_propagation_iterations=8,
                unsupported_propagation_neighbor_weight=0.75,
                smoothing_iterations=8,
            ),
        },
    ),
)


def get_method_spec(token: str) -> MethodSpec:
    for spec in METHOD_SPECS:
        if spec.matches(token):
            return spec
    raise KeyError(f"Unknown method token: {token}")


def get_method_specs(tokens: Iterable[str] | None = None) -> list[MethodSpec]:
    if tokens is None:
        return list(METHOD_SPECS)
    return [get_method_spec(token) for token in tokens]


def get_dynamic_method_display_names() -> list[str]:
    return [spec.display_name for spec in METHOD_SPECS if spec.execution_family == "dynamic"]


def get_runnable_method_display_names() -> list[str]:
    return [spec.display_name for spec in METHOD_SPECS]


def get_default_main_table_methods() -> list[MethodSpec]:
    return [spec for spec in METHOD_SPECS if spec.paper_role in {"external-baseline", "ours"}]


def get_external_baseline_methods() -> list[MethodSpec]:
    return [spec for spec in METHOD_SPECS if spec.is_external and spec.paper_role == "external-baseline"]


def get_supplementary_baseline_methods() -> list[MethodSpec]:
    return [spec for spec in METHOD_SPECS if spec.paper_role == "supplementary-baseline"]


def profile_cli_args(spec: MethodSpec, profile_name: str | None) -> list[str]:
    if not profile_name:
        return []
    profile = spec.default_profiles.get(profile_name)
    if profile is None:
        return []

    args: list[str] = []
    for key, value in profile.cli_overrides.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            else:
                args.append(f"--no-{key.replace('_', '-')}")
            continue
        args.extend([flag, str(value)])
    return args
