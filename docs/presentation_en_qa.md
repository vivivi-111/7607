# English PPTX QA report

- Deliverable: `docs/presentation_en.pptx`
- Slides: 14
- Embedded figure assets: 8
- Slides with speaker notes: 14/14
- Source deck script: `docs/presentation_en_script.md`
- Build script: `scripts/build_presentation_en.py`

## Checks completed

- Reopened successfully with `python-pptx`.
- Nature Paper-to-PPTX audit completed with `high=0`, `medium=0`.
- All slide objects remain inside the 16:9 slide canvas.
- No placeholder text, clipped scientific crop, or missing source figure was detected.
- Figure slides use the original publication-ready PNG files without data alteration.
- Speaker notes were embedded in the PPTX and separately exported as Markdown.
- Slide compositions vary across claim-led, process-wide, figure-dominant,
  comparison, audit, limitation, and synthesis layouts.
- On-slide text is intentionally shorter than the speaker notes.

## Corrective review

The first generated deck produced no high- or medium-severity findings. The
remaining audit messages are low-severity near-alignment heuristics caused by
deliberate inset text within aligned panels; they do not indicate overlap or
out-of-bounds content.

## Known limitation

LibreOffice/PowerPoint rendering was unavailable in the execution environment,
so QA used source-figure inspection, PPTX package reopening, shape-bound checks,
speaker-note counts, and the XML quality audit rather than rendered slide
previews.
