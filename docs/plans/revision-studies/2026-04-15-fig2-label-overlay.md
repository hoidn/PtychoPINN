# Fig 2 Label Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Fig. 2 labels readable by preserving the original PNGs, generating derived textless PNG assets, and rebuilding every baked-in label/tick/title as larger LaTeX/TikZ text.

**Architecture:** Keep the scientific raster content from the current Fig. 2 source images, but move typography ownership to the manuscript. A small paper-side Python script creates `*_textless.png` derivatives by copying only image panels and colorbar bars onto a white canvas, never overwriting `lowcounts.png` or `8192.png`. `ptychopinn_2025.tex` then renders those textless assets inside inline TikZ overlays with manuscript-scale fonts.

**Tech Stack:** PATH `python`, Pillow, `pytest`/`unittest`, LaTeX/TikZ, `pdflatex`, `pdftoppm`, ImageMagick `identify`.

---

## Brainstorming Outcome

Current Fig. 2 is `fig:smalldat` in `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex`. It includes two baked-label PNGs:

- `/home/ollie/Documents/ptychopinnpaper2/figures/lowcounts.png` (`2984x1441`)
- `/home/ollie/Documents/ptychopinnpaper2/figures/8192.png` (`2048x999`)

The request is to replace **all** baked-in text, not only the smallest labels. That includes the global title, model/panel headings, row labels, colorbar labels, and colorbar numeric tick labels. The original source PNGs must remain untouched.

### Approaches Considered

**Recommended: derived textless PNGs + inline LaTeX/TikZ overlays.**  
This preserves the accepted scientific image pixels while making typography editable and consistent with the paper. It also keeps original PNGs intact and makes the change reviewable: generated textless assets plus a manifest document exactly what was kept.

**Alternative: regenerate Fig. 2 from original arrays.**  
This would be cleaner scientifically, but current provenance says `8192.png` has no byte-identical recovered source. Regenerating from data risks changing image content while trying to fix only typography.

**Alternative: bake larger labels into new PNGs.**  
This is faster but repeats the current problem: labels remain rasterized, hard to tune in the manuscript, and less consistent with other LaTeX-rendered figure text.

### Design Decision

Use current `lowcounts.png` and `8192.png` only as raster sources. Generate `lowcounts_textless.png` and `8192_textless.png` by pasting panel/colorbar regions onto a blank white canvas. Rebuild all text in inline TikZ within `ptychopinn_2025.tex`; do **not** use `\input{...}` because the manuscript template explicitly requests a single TeX file.

---

## File Structure

- Create: `/home/ollie/Documents/ptychopinnpaper2/figures/scripts/fig2_smalldat_textless.py`
  - Responsibility: generate textless derived assets and a manifest from current Fig. 2 PNGs.
- Create: `/home/ollie/Documents/ptychopinnpaper2/figures/scripts/test_fig2_smalldat_textless.py`
  - Responsibility: verify the script preserves sources, refuses output/source collisions, pastes only requested keep regions, and writes provenance.
- Create: `/home/ollie/Documents/ptychopinnpaper2/figures/lowcounts_textless.png`
  - Generated textless source for Fig. 2a overlay.
- Create: `/home/ollie/Documents/ptychopinnpaper2/figures/8192_textless.png`
  - Generated textless source for Fig. 2b overlay.
- Create: `/home/ollie/Documents/ptychopinnpaper2/figures/fig2_smalldat_textless_manifest.json`
  - Records source/output paths, dimensions, hashes, keep rectangles, and no-overwrite policy.
- Modify: `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex`
  - Replace direct `\includegraphics` calls for Fig. 2 with inline TikZ overlays.
- Modify: `/home/ollie/Documents/ptychopinnpaper2/data/README.md`
  - Record derived Fig. 2 assets and provenance.
- Modify: `/home/ollie/Documents/ptychopinnpaper2/changelog.txt`
  - Add reviewer-facing note about Fig. 2 typography.
- Modify: `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`
  - Add progress/verification note under the compile/readability verification item.

---

### Task 1: Add Failing Tests For Textless Asset Generation

**Files:**
- Create: `/home/ollie/Documents/ptychopinnpaper2/figures/scripts/test_fig2_smalldat_textless.py`

- [ ] **Step 1: Write failing tests**

Use the existing Fig. 1 figure-script test style: `unittest`, `tempfile`, `Path`, and Pillow.

```python
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import fig2_smalldat_textless as fig2


class Fig2SmalldatTextlessTests(unittest.TestCase):
    def test_sha256_file_is_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            Image.new("RGB", (4, 4), (10, 20, 30)).save(path)

            self.assertEqual(fig2.sha256_file(path), fig2.sha256_file(path))
            self.assertEqual(len(fig2.sha256_file(path)), 64)

    def test_refuses_to_overwrite_source_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.png"
            Image.new("RGB", (4, 4), "white").save(source)
            spec = fig2.FigureTextlessSpec(
                figure_id="fixture",
                source_path=source,
                output_path=source,
                expected_dimensions=(4, 4),
                keep_rectangles=(fig2.KeepRectangle("panel", 0, 0, 2, 2),),
            )

            with self.assertRaises(ValueError):
                fig2.validate_spec(spec)

    def test_apply_keep_rectangles_preserves_only_requested_pixels(self):
        image = Image.new("RGB", (8, 8), "white")
        for x in range(2, 6):
            for y in range(2, 6):
                image.putpixel((x, y), (1, 2, 3))
        image.putpixel((0, 0), (200, 0, 0))

        output = fig2.apply_keep_rectangles(
            image,
            (fig2.KeepRectangle("panel", 2, 2, 6, 6),),
        )

        self.assertEqual(output.getpixel((3, 3)), (1, 2, 3))
        self.assertEqual(output.getpixel((0, 0)), (255, 255, 255))

    def test_write_manifest_records_source_and_output_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.png"
            output = tmp_path / "output.png"
            manifest = tmp_path / "manifest.json"
            Image.new("RGB", (4, 4), (1, 2, 3)).save(source)
            Image.new("RGB", (4, 4), "white").save(output)
            spec = fig2.FigureTextlessSpec(
                figure_id="fixture",
                source_path=source,
                output_path=output,
                expected_dimensions=(4, 4),
                keep_rectangles=(fig2.KeepRectangle("panel", 0, 0, 2, 2),),
            )

            fig2.write_manifest(manifest, [spec], command=["fixture"])

            payload = json.loads(manifest.read_text())
            self.assertEqual(payload["schema"], "fig2_smalldat_textless.v1")
            self.assertEqual(payload["figures"][0]["source"]["sha256"], fig2.sha256_file(source))
            self.assertEqual(payload["figures"][0]["output"]["sha256"], fig2.sha256_file(output))
            self.assertTrue(payload["no_original_overwrite"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/ollie/Documents/ptychopinnpaper2
PYTHONPATH=figures/scripts python -m pytest figures/scripts/test_fig2_smalldat_textless.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'fig2_smalldat_textless'`.

- [ ] **Step 3: Commit tests if working in commit-sized increments**

```bash
cd /home/ollie/Documents/ptychopinnpaper2
git add figures/scripts/test_fig2_smalldat_textless.py
git commit -m "test(paper): add fig2 textless asset tests"
```

---

### Task 2: Implement Textless Asset Generator

**Files:**
- Create: `/home/ollie/Documents/ptychopinnpaper2/figures/scripts/fig2_smalldat_textless.py`

- [ ] **Step 1: Implement the generator**

Use Pillow to paste only the scientific panel/colorbar rectangles onto a white canvas. Start with manually calibrated keep rectangles from the current source PNGs, then tune in Task 3 if visual inspection shows clipping or leftover labels.

```python
#!/usr/bin/env python3
"""Generate textless derived assets for manuscript Fig. 2."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "fig2_smalldat_textless_manifest.json"


@dataclass(frozen=True)
class KeepRectangle:
    label: str
    left: int
    top: int
    right: int
    bottom: int

    def as_payload(self) -> dict[str, int | str]:
        return {
            "label": self.label,
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
        }


@dataclass(frozen=True)
class FigureTextlessSpec:
    figure_id: str
    source_path: Path
    output_path: Path
    expected_dimensions: tuple[int, int]
    keep_rectangles: tuple[KeepRectangle, ...]


LOWCOUNTS_RECTS = (
    KeepRectangle("phase_ptychopinn", 50, 200, 590, 740),
    KeepRectangle("phase_ptychopinn_colorbar", 610, 200, 640, 740),
    KeepRectangle("phase_baseline", 795, 200, 1335, 740),
    KeepRectangle("phase_baseline_colorbar", 1355, 200, 1385, 740),
    KeepRectangle("phase_tike", 1540, 175, 2130, 765),
    KeepRectangle("phase_ground_truth", 2285, 200, 2825, 740),
    KeepRectangle("phase_ground_truth_colorbar", 2845, 200, 2875, 740),
    KeepRectangle("amplitude_ptychopinn", 50, 805, 590, 1340),
    KeepRectangle("amplitude_ptychopinn_colorbar", 610, 805, 640, 1340),
    KeepRectangle("amplitude_baseline", 795, 805, 1335, 1340),
    KeepRectangle("amplitude_baseline_colorbar", 1355, 805, 1385, 1340),
    KeepRectangle("amplitude_tike", 1540, 835, 2130, 1370),
    KeepRectangle("amplitude_ground_truth", 2285, 805, 2825, 1340),
    KeepRectangle("amplitude_ground_truth_colorbar", 2845, 805, 2875, 1340),
)

PATTERNS_8192_RECTS = (
    KeepRectangle("phase_ptychopinn", 35, 100, 420, 398),
    KeepRectangle("phase_ptychopinn_colorbar", 438, 100, 458, 398),
    KeepRectangle("phase_baseline", 535, 100, 920, 398),
    KeepRectangle("phase_baseline_colorbar", 938, 100, 958, 398),
    KeepRectangle("phase_tike", 1038, 85, 1460, 414),
    KeepRectangle("phase_ground_truth", 1540, 100, 1925, 398),
    KeepRectangle("phase_ground_truth_colorbar", 1944, 100, 1964, 398),
    KeepRectangle("amplitude_ptychopinn", 35, 452, 420, 748),
    KeepRectangle("amplitude_ptychopinn_colorbar", 438, 452, 458, 748),
    KeepRectangle("amplitude_baseline", 535, 452, 920, 748),
    KeepRectangle("amplitude_baseline_colorbar", 938, 452, 958, 748),
    KeepRectangle("amplitude_tike", 1038, 438, 1460, 764),
    KeepRectangle("amplitude_ground_truth", 1540, 452, 1925, 748),
    KeepRectangle("amplitude_ground_truth_colorbar", 1944, 452, 1964, 748),
)

FIGURE_SPECS = (
    FigureTextlessSpec(
        figure_id="lowcounts",
        source_path=REPO_ROOT / "lowcounts.png",
        output_path=REPO_ROOT / "lowcounts_textless.png",
        expected_dimensions=(2984, 1441),
        keep_rectangles=LOWCOUNTS_RECTS,
    ),
    FigureTextlessSpec(
        figure_id="8192",
        source_path=REPO_ROOT / "8192.png",
        output_path=REPO_ROOT / "8192_textless.png",
        expected_dimensions=(2048, 999),
        keep_rectangles=PATTERNS_8192_RECTS,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def validate_spec(spec: FigureTextlessSpec) -> None:
    if spec.source_path.resolve() == spec.output_path.resolve():
        raise ValueError(f"{spec.figure_id}: output would overwrite source {spec.source_path}")
    if not spec.source_path.exists():
        raise FileNotFoundError(spec.source_path)
    dimensions = image_dimensions(spec.source_path)
    if dimensions != spec.expected_dimensions:
        raise ValueError(
            f"{spec.figure_id}: expected {spec.expected_dimensions}, got {dimensions}; "
            "recalibrate keep rectangles before generating Fig. 2 textless assets"
        )


def apply_keep_rectangles(image: Image.Image, rectangles: Iterable[KeepRectangle]) -> Image.Image:
    source = image.convert("RGB")
    output = Image.new("RGB", source.size, "white")
    for rectangle in rectangles:
        box = (rectangle.left, rectangle.top, rectangle.right, rectangle.bottom)
        output.paste(source.crop(box), box)
    return output


def generate_textless(spec: FigureTextlessSpec) -> None:
    validate_spec(spec)
    with Image.open(spec.source_path) as image:
        output = apply_keep_rectangles(image, spec.keep_rectangles)
    output.save(spec.output_path)


def write_manifest(path: Path, specs: Iterable[FigureTextlessSpec], command: list[str]) -> None:
    payload = {
        "schema": "fig2_smalldat_textless.v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": command,
        "no_original_overwrite": True,
        "figures": [],
    }
    for spec in specs:
        payload["figures"].append(
            {
                "figure_id": spec.figure_id,
                "source": {
                    "path": str(spec.source_path),
                    "dimensions_px": list(image_dimensions(spec.source_path)),
                    "sha256": sha256_file(spec.source_path),
                },
                "output": {
                    "path": str(spec.output_path),
                    "dimensions_px": list(image_dimensions(spec.output_path)),
                    "sha256": sha256_file(spec.output_path),
                },
                "keep_rectangles": [rect.as_payload() for rect in spec.keep_rectangles],
            }
        )
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for spec in FIGURE_SPECS:
        generate_textless(spec)
    write_manifest(args.manifest, FIGURE_SPECS, command=["figures/scripts/fig2_smalldat_textless.py"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run tests**

Run:

```bash
cd /home/ollie/Documents/ptychopinnpaper2
PYTHONPATH=figures/scripts python -m pytest figures/scripts/test_fig2_smalldat_textless.py -q
```

Expected: PASS.

- [ ] **Step 3: Commit implementation if working in commit-sized increments**

```bash
cd /home/ollie/Documents/ptychopinnpaper2
git add figures/scripts/fig2_smalldat_textless.py figures/scripts/test_fig2_smalldat_textless.py
git commit -m "feat(paper): generate fig2 textless assets"
```

---

### Task 3: Generate And Calibrate Textless PNG Assets

**Files:**
- Create: `/home/ollie/Documents/ptychopinnpaper2/figures/lowcounts_textless.png`
- Create: `/home/ollie/Documents/ptychopinnpaper2/figures/8192_textless.png`
- Create: `/home/ollie/Documents/ptychopinnpaper2/figures/fig2_smalldat_textless_manifest.json`
- Modify if needed: `/home/ollie/Documents/ptychopinnpaper2/figures/scripts/fig2_smalldat_textless.py`

- [ ] **Step 1: Capture original hashes**

Run:

```bash
cd /home/ollie/Documents/ptychopinnpaper2
sha256sum figures/lowcounts.png figures/8192.png
```

Expected: record both hashes in the task notes before generation.

- [ ] **Step 2: Generate textless assets**

Run:

```bash
cd /home/ollie/Documents/ptychopinnpaper2
python figures/scripts/fig2_smalldat_textless.py
identify figures/lowcounts_textless.png figures/8192_textless.png
```

Expected:

- `figures/lowcounts_textless.png PNG 2984x1441 ...`
- `figures/8192_textless.png PNG 2048x999 ...`
- `figures/fig2_smalldat_textless_manifest.json` exists and records original/output hashes.

- [ ] **Step 3: Verify originals were not overwritten**

Run:

```bash
cd /home/ollie/Documents/ptychopinnpaper2
sha256sum figures/lowcounts.png figures/8192.png
```

Expected: hashes exactly match Step 1.

- [ ] **Step 4: Inspect generated textless assets**

Open these local images with the image viewer or use the `view_image` tool:

- `/home/ollie/Documents/ptychopinnpaper2/figures/lowcounts_textless.png`
- `/home/ollie/Documents/ptychopinnpaper2/figures/8192_textless.png`

Expected:

- all baked title/row/column/colorbar text is gone;
- image panels, colorbar bars, and non-text scientific annotations remain;
- no panel data is clipped;
- source PNGs remain unchanged.

- [ ] **Step 5: Calibrate keep rectangles if needed**

If any baked text remains or panel/colorbar pixels are clipped, adjust only the affected `KeepRectangle` constants in `fig2_smalldat_textless.py`, regenerate, and repeat Steps 2-4.

- [ ] **Step 6: Commit generated assets if working in commit-sized increments**

```bash
cd /home/ollie/Documents/ptychopinnpaper2
git add figures/lowcounts_textless.png figures/8192_textless.png figures/fig2_smalldat_textless_manifest.json figures/scripts/fig2_smalldat_textless.py
git commit -m "build(paper): add fig2 textless derived assets"
```

---

### Task 4: Replace Fig. 2 Direct Includes With Inline TikZ Overlays

**Files:**
- Modify: `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex`

- [ ] **Step 1: Add Fig. 2 font macros near existing color definitions**

Add after the existing `\definecolor{...}` block:

```tex
% Fig. 2 overlay typography. Labels are rendered in LaTeX because the
% original image text was too small when rasterized.
\newcommand{\figtwotitlefs}{\small\bfseries}
\newcommand{\figtwocolfs}{\footnotesize\bfseries}
\newcommand{\figtworowfs}{\footnotesize}
\newcommand{\figtwocbarfs}{\scriptsize}
```

- [ ] **Step 2: Replace the first Fig. 2 subfigure image with a TikZ overlay**

Replace:

```tex
\includegraphics[width=0.8\textwidth]{figures/lowcounts.png}
```

with:

```tex
\begin{tikzpicture}
  \node[anchor=south west, inner sep=0] (figtwoa) at (0,0)
    {\includegraphics[width=0.8\textwidth]{figures/lowcounts_textless.png}};
  \begin{scope}[x={(figtwoa.south east)}, y={(figtwoa.north west)}]
    \node[font=\figtwotitlefs, anchor=north] at (0.50,0.995)
      {PtychoPINN vs. Baseline vs. Tike Reconstruction};

    \node[font=\figtwocolfs, anchor=south] at (0.115,0.865) {PtychoPINN};
    \node[font=\figtwocolfs, anchor=south] at (0.380,0.865) {Baseline};
    \node[font=\figtwocolfs, anchor=south] at (0.660,0.875) {Tike};
    \node[font=\figtwocolfs, anchor=south] at (0.860,0.865) {Ground Truth};

    \node[font=\figtworowfs, rotate=90, anchor=center] at (0.012,0.625) {Phase};
    \node[font=\figtworowfs, rotate=90, anchor=center] at (0.012,0.210) {Amplitude};

    % Colorbar labels and tick values. Coordinates should be tuned after the
    % first compiled-PDF inspection.
    \node[font=\figtwocbarfs, rotate=90, anchor=center] at (0.250,0.625) {Phase (rad)};
    \node[font=\figtwocbarfs, rotate=90, anchor=center] at (0.515,0.625) {Phase (rad)};
    \node[font=\figtwocbarfs, rotate=90, anchor=center] at (0.985,0.625) {Phase (rad)};
    \node[font=\figtwocbarfs, rotate=90, anchor=center] at (0.250,0.210) {Amplitude};
    \node[font=\figtwocbarfs, rotate=90, anchor=center] at (0.515,0.210) {Amplitude};
    \node[font=\figtwocbarfs, rotate=90, anchor=center] at (0.985,0.210) {Amplitude};
  \end{scope}
\end{tikzpicture}
```

Then add numeric colorbar ticks in the same scope. Use the original visible tick values as the source of truth and tune positions after rendering. Do not omit numeric tick labels unless a later human review explicitly approves a less dense colorbar.

- [ ] **Step 3: Replace the second Fig. 2 subfigure image with a TikZ overlay**

Replace:

```tex
\includegraphics[width=0.8\textwidth]{figures/8192.png}
```

with:

```tex
\begin{tikzpicture}
  \node[anchor=south west, inner sep=0] (figtwob) at (0,0)
    {\includegraphics[width=0.8\textwidth]{figures/8192_textless.png}};
  \begin{scope}[x={(figtwob.south east)}, y={(figtwob.north west)}]
    \node[font=\figtwotitlefs, anchor=north] at (0.50,0.995)
      {PtychoPINN vs. Baseline vs. Tike Reconstruction};

    \node[font=\figtwocolfs, anchor=south] at (0.090,0.895) {PtychoPINN};
    \node[font=\figtwocolfs, anchor=south] at (0.320,0.895) {Baseline};
    \node[font=\figtwocolfs, anchor=south] at (0.552,0.905) {Tike};
    \node[font=\figtwocolfs, anchor=south] at (0.790,0.895) {Ground Truth};

    \node[font=\figtworowfs, rotate=90, anchor=center] at (0.012,0.650) {Phase};
    \node[font=\figtworowfs, rotate=90, anchor=center] at (0.012,0.240) {Amplitude};

    \node[font=\figtwocbarfs, rotate=90, anchor=center] at (0.245,0.650) {Phase (rad)};
    \node[font=\figtwocbarfs, rotate=90, anchor=center] at (0.490,0.650) {Phase (rad)};
    \node[font=\figtwocbarfs, rotate=90, anchor=center] at (0.985,0.650) {Phase (rad)};
    \node[font=\figtwocbarfs, rotate=90, anchor=center] at (0.245,0.240) {Amplitude};
    \node[font=\figtwocbarfs, rotate=90, anchor=center] at (0.490,0.240) {Amplitude};
    \node[font=\figtwocbarfs, rotate=90, anchor=center] at (0.985,0.240) {Amplitude};
  \end{scope}
\end{tikzpicture}
```

Then add numeric colorbar ticks in the same scope. Use the original visible tick values as the source of truth and tune positions after rendering.

- [ ] **Step 4: Compile once to catch syntax errors**

Run:

```bash
cd /home/ollie/Documents/ptychopinnpaper2
pdflatex -interaction=nonstopmode -halt-on-error ptychopinn_2025.tex
```

Expected: exit code `0`.

- [ ] **Step 5: Tune overlay coordinates**

Render and inspect the Fig. 2 page. Adjust only TikZ coordinates/font macros until:

- no overlay text overlaps image panels or colorbar bars;
- no baked text remains visible;
- labels are readable at manuscript scale;
- Fig. 2 still fits the page and keeps its two subcaptions.

- [ ] **Step 6: Commit LaTeX overlay if working in commit-sized increments**

```bash
cd /home/ollie/Documents/ptychopinnpaper2
git add ptychopinn_2025.tex
git commit -m "fix(paper): render fig2 labels in latex"
```

---

### Task 5: Update Provenance And Revision Notes

**Files:**
- Modify: `/home/ollie/Documents/ptychopinnpaper2/data/README.md`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/changelog.txt`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`

- [ ] **Step 1: Update `data/README.md` Fig. 2 section**

Under `## Small-Data Reconstructions (fig:smalldat)`, add:

```markdown
Typography update:
- Original source rasters are preserved:
  - `paper/figures/lowcounts.png`
  - `paper/figures/8192.png`
- The manuscript uses derived textless assets:
  - `paper/figures/lowcounts_textless.png`
  - `paper/figures/8192_textless.png`
- Derived asset provenance:
  - `paper/figures/fig2_smalldat_textless_manifest.json`
- All title, row/column, colorbar, and numeric tick labels are rendered in LaTeX/TikZ in `ptychopinn_2025.tex` so the font size can be tuned at manuscript scale.
```

- [ ] **Step 2: Update `changelog.txt`**

Add a concise entry:

```text
- Rebuilt Fig. 2 typography by preserving the original small-data PNGs, generating textless derived assets, and rendering all figure labels/ticks in LaTeX/TikZ with larger manuscript-scale fonts.
```

- [ ] **Step 3: Update `reviewer_revision_checklist.md`**

Under the verification/readability item, add a dated note:

```markdown
  Update (2026-04-15): Fig. 2 labels were moved from baked raster text into LaTeX/TikZ overlays using derived textless PNG assets; original `lowcounts.png` and `8192.png` were preserved unchanged. The compiled PDF page was inspected for label readability.
```

- [ ] **Step 4: Commit provenance notes if working in commit-sized increments**

```bash
cd /home/ollie/Documents/ptychopinnpaper2
git add data/README.md changelog.txt reviewer_revision_checklist.md
git commit -m "docs(paper): record fig2 label overlay provenance"
```

---

### Task 6: Verify Paper Build And Visual Output

**Files:**
- Read/verify: `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.pdf`
- Create artifact directory: `/home/ollie/Documents/ptychopinnpaper2/artifacts/work/fig2-label-overlay/`

- [ ] **Step 1: Run tests**

Run:

```bash
cd /home/ollie/Documents/ptychopinnpaper2
PYTHONPATH=figures/scripts python -m pytest figures/scripts/test_fig2_smalldat_textless.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Verify generated assets and source hashes**

Run:

```bash
cd /home/ollie/Documents/ptychopinnpaper2
identify figures/lowcounts.png figures/lowcounts_textless.png figures/8192.png figures/8192_textless.png
sha256sum figures/lowcounts.png figures/8192.png
```

Expected:

- original dimensions unchanged;
- textless dimensions match source dimensions;
- original hashes match the hashes captured before generation and those recorded in the manifest.

- [ ] **Step 3: Compile the manuscript twice**

Run:

```bash
cd /home/ollie/Documents/ptychopinnpaper2
pdflatex -interaction=nonstopmode -halt-on-error ptychopinn_2025.tex
pdflatex -interaction=nonstopmode -halt-on-error ptychopinn_2025.tex
```

Expected: both commands exit `0`.

- [ ] **Step 4: Render the Fig. 2 PDF page for inspection**

The current aux file places Fig. 2 on page 7. Confirm after compiling, then render:

```bash
cd /home/ollie/Documents/ptychopinnpaper2
mkdir -p artifacts/work/fig2-label-overlay
pdftoppm -f 7 -l 7 -r 200 -png ptychopinn_2025.pdf artifacts/work/fig2-label-overlay/ptychopinn_2025_page
```

Expected: `artifacts/work/fig2-label-overlay/ptychopinn_2025_page-7.png` exists. If Fig. 2 moved, use the correct page number.

- [ ] **Step 5: Inspect the rendered page**

Open the rendered page locally or with the `view_image` tool.

Acceptance criteria:

- all Fig. 2 baked text is gone from the image rasters;
- all replacement text is LaTeX-rendered and noticeably larger than before;
- colorbar labels and numeric ticks are readable;
- no text overlaps data panels or colorbars;
- no source scientific image panel is clipped or shifted;
- original `lowcounts.png` and `8192.png` are unchanged.

- [ ] **Step 6: Run stale-reference scans**

Run:

```bash
cd /home/ollie/Documents/ptychopinnpaper2
rg -n "lowcounts_textless|8192_textless|fig2_smalldat_textless" ptychopinn_2025.tex data/README.md changelog.txt reviewer_revision_checklist.md figures/fig2_smalldat_textless_manifest.json
rg -n "\\\\includegraphics\\[width=0\\.8\\\\textwidth\\]\\{figures/(lowcounts|8192)\\.png\\}" ptychopinn_2025.tex
```

Expected:

- first command finds the new overlay assets/provenance;
- second command returns no matches because Fig. 2 should no longer include the labeled source PNGs directly.

- [ ] **Step 7: Final commit if requested**

```bash
cd /home/ollie/Documents/ptychopinnpaper2
git add figures/scripts/fig2_smalldat_textless.py \
        figures/scripts/test_fig2_smalldat_textless.py \
        figures/lowcounts_textless.png \
        figures/8192_textless.png \
        figures/fig2_smalldat_textless_manifest.json \
        ptychopinn_2025.tex \
        data/README.md \
        changelog.txt \
        reviewer_revision_checklist.md
git commit -m "fix(paper): improve fig2 label readability"
```

---

## Notes For Executor

- Do not overwrite `/home/ollie/Documents/ptychopinnpaper2/figures/lowcounts.png`.
- Do not overwrite `/home/ollie/Documents/ptychopinnpaper2/figures/8192.png`.
- Keep all TikZ overlay code inline in `ptychopinn_2025.tex`; do not use `\input{...}` for the figure.
- Use the original visible tick values as the source of truth for replacement numeric colorbar labels.
- If the full colorbar tick set becomes unreadable at larger font size, stop and ask for a presentation decision rather than silently dropping tick labels.
- Existing uncommitted work is present in the paper checkout. Do not revert or overwrite unrelated changes.
