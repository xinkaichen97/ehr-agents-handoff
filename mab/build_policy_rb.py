"""S2: give the actor the duplicate policy and the same chart orders the read-back auditor sees.

Writes duplicates_v1_policy_rb.json from duplicates_v1_policy.json. Run it with --arm none.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_mab as m  # noqa: E402

here = Path(__file__).parent
cases = json.loads((here / "duplicates_v1_policy.json").read_text())
for c in cases:
    # Same listing and wording as auditor_review(read_back=True).
    listing = "\n".join(f"{when}  {text}" for when, text in m.chart_orders(c["eval_MRN"])[:100]) or "none"
    c["context"] += f"\n\nMedication orders read from the chart, newest first:\n{listing}"
(here / "duplicates_v1_policy_rb.json").write_text(json.dumps(cases, indent=1))
print(len(cases), "cases; mean context chars", sum(len(c["context"]) for c in cases) // len(cases))
