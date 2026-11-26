# Paper source

This directory uses the official NeurIPS 2026 LaTeX style supplied by the
author as a layout template. The manuscript is not currently intended as a
NeurIPS submission, so the NeurIPS Paper Checklist is intentionally omitted.

## Files

- `main.tex`: anonymized manuscript.
- `neurips_2026.sty`: unmodified official style file.
- `neurips_2026.tex`: original template source file.
- `figures/`: publication figures, source-data CSVs, plotting scripts, and QA
  artifacts.

## Build

```bash
tectonic -X compile main.tex
```

The manuscript is organized as requested:

1. Abstract
2. Introduction
3. Related work
4. Methods
5. Experiments and evaluation
6. Discussion, limitations, future work, and broader impact
7. Conclusion

The supplemental repository is MIT licensed and includes an asset-license
inventory. The source archive omits model weights and preserves anonymity.
Measured hardware and DPO training times are reported in the reproducibility
appendix.
