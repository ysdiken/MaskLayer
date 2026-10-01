"""
Admin endpoints — deployment configuration.

GET /api/v1/admin/mask-policy
    Returns the entity catalog (label, Turkish display, detector group) and the
    current enabled/disabled state of each type.

PUT /api/v1/admin/mask-policy
    Updates the enabled/disabled state for one or more entity types.

The mask policy is a post-detection OUTPUT filter (see masking/mask_policy.py):
it controls which detected entity types are rendered as placeholders, not what
the model detects. Disabling a type never affects detection, the corrections
log, or the training-examples store.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.masking import mask_policy

router = APIRouter()


class MaskPolicyResponse(BaseModel):
    # Catalog entries: {label, display, group} — drives the admin UI.
    labels: list[dict[str, str]]
    # label → enabled (True = masked in output).
    policy: dict[str, bool]


class MaskPolicyUpdate(BaseModel):
    # Partial map: only the labels being changed need to be present.
    policy: dict[str, bool] = Field(default_factory=dict)


@router.get("/api/v1/admin/mask-policy", response_model=MaskPolicyResponse)
async def get_mask_policy() -> MaskPolicyResponse:
    return MaskPolicyResponse(
        labels=mask_policy.MASK_LABELS,
        policy=mask_policy.get_policy(),
    )


@router.put("/api/v1/admin/mask-policy", response_model=MaskPolicyResponse)
async def update_mask_policy(update: MaskPolicyUpdate) -> MaskPolicyResponse:
    new_policy = mask_policy.update_policy(update.policy)
    return MaskPolicyResponse(
        labels=mask_policy.MASK_LABELS,
        policy=new_policy,
    )
