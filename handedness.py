"""
handedness.py
=============
Subject handedness metadata — reference only.

KEY FINDING FROM DATA (left-hander analysis, subjects 4/9/15/17):
  Standard whole-scalp PLV helped ALL 4 left-handed subjects.
  The initial hypothesis (PLV hurts left-handers due to mixed hemispheric
  dominance) was not supported by this dataset.

  Likely reason: visual imagery of Arabic letters is a bilateral cognitive
  task. Both hemispheres engage regardless of handedness, so whole-scalp
  PLV captures real signal for everyone.

PIPELINE CONSEQUENCE:
  All subjects — left- or right-handed — use the same feature combo:
      Riem + Band Power + PLV  (+ adaptive CSP)
  Handedness is no longer a branching condition in the pipeline.

This file is kept for:
  - Reporting / logging which subjects are left-handed
  - Future research that may find handedness effects on other features
"""

# ── Ground truth (Table 1 of the paper) ──────────────────────────────────────
LEFT_HANDED_SUBJECTS  = {4, 9, 15, 17}
RIGHT_HANDED_SUBJECTS = set(range(1, 31)) - LEFT_HANDED_SUBJECTS


def is_left_handed(subject_id: int) -> bool:
    return subject_id in LEFT_HANDED_SUBJECTS


def get_handedness(subject_id: int) -> str:
    return "left" if is_left_handed(subject_id) else "right"


def print_subject_handedness_table():
    """Pretty-print handedness for all 30 subjects."""
    print("\n  Subject Handedness Table")
    print("  " + "─" * 35)
    print(f"  {'Subject':<10} {'Hand':<10} {'Note'}")
    print("  " + "─" * 35)
    for sid in range(1, 31):
        hand   = get_handedness(sid)
        marker = " ◄ left-handed" if is_left_handed(sid) else ""
        print(f"  S{sid:02d}       {hand:<10}{marker}")
    print(f"\n  Left-handed  : {sorted(LEFT_HANDED_SUBJECTS)}")
    print(f"  Right-handed : {len(RIGHT_HANDED_SUBJECTS)} subjects")
    print(
        "\n  Note: PLV strategy is identical for all subjects (StdPLV).\n"
        "  Handedness metadata is retained for reporting purposes only."
    )