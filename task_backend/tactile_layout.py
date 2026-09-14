"""One-based glove channel positions, fabric electronic skin manual pp. 11–13.

Transport has 256 ADC slots; the published map identifies 132 pressure points
and five bend channels. The manual's advertised 162 points are not explained.
Rows follow the manual's palm-facing view, left to right and top to bottom.
"""

FINGER_NAMES = ("拇指", "食指", "中指", "无名指", "小指")
FINGERS = {
    "right": [
        [240,239,238,256,255,254,16,15,14,32,31,30],
        [237,236,235,253,252,251,13,12,11,29,28,27],
        [234,233,232,250,249,248,10,9,8,26,25,24],
        [231,230,229,247,246,245,7,6,5,23,22,21],
        [228,227,226,244,243,242,4,3,2,20,19,18],
    ],
    "left": [
        [19,18,17,3,2,1,243,242,241,227,226,225],
        [22,21,20,6,5,4,246,245,244,230,229,228],
        [25,24,23,9,8,7,249,248,247,233,232,231],
        [28,27,26,12,11,10,252,251,250,236,235,234],
        [31,30,29,15,14,13,255,254,253,239,238,237],
    ],
}
PALM_ROWS = {
    "right": [list(range(61,49,-1))] + [list(range(n,n-15,-1)) for n in (80,96,112,128)],
    "left": [list(range(207,195,-1))] + [list(range(n,n-15,-1)) for n in (191,175,159,143)],
}
BEND_IDS = {"right": [47,44,41,38,35], "left": [210,213,216,219,222]}


def glove_layout(side):
    if side not in FINGERS:
        return None
    return {
        "fingers": [{"label": name, "ids": ids, "bend_id": bend}
                    for name, ids, bend in zip(FINGER_NAMES, FINGERS[side], BEND_IDS[side])],
        "palm_rows": PALM_ROWS[side],
        "palm_first_offset": 3 if side == "right" else 0,
        "pressure_count": 132,
        "bend_count": 5,
    }
