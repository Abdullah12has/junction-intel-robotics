# Station classification rule — 14 November 2026

User-provided rule: rectangular bricks are defective; square bricks are good. This supersedes earlier assumptions about surface blemishes or defect-bearing faces. Route square/good to the green bowl and rectangular/defective to the red bowl, consistent with the existing task strings.

Collect normal PatchCore training images of square bricks, and validation/held-out examples of both shapes. Include different colors and orientations so shape is evaluated rather than accidental color/background correlations. Mount the inspection camera so the outline is visible at the fixed inspection spot. Camera position, ROI, exposure and HOME remain unverified.

Still unanswered: judging sequence and final rubric. The user supplied the classification rule; it is not independently verified mentor correspondence.

Git preference: keep commits local for now; do not retry publishing without new authorization.
