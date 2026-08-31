"""Pure-Python SMPL-X MANO hand and joint-order contract."""

MANO_HAND_ORDER = ("left", "right")

SMPLX_MANO_PARENT_INDICES = (
    -1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 15, 3, 6, 12, 9,
)

SMPLX_MANO_JOINT_NAMES = (
    "wrist",
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "pinky_mcp", "pinky_pip", "pinky_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "thumb_mcp", "thumb_pip", "thumb_dip",
    "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip",
)

SMPLX_MANO_SKELETON_EDGES = tuple(
    (parent, joint)
    for joint, parent in enumerate(SMPLX_MANO_PARENT_INDICES)
    if parent >= 0
)

