#!/usr/bin/env python3
"""Write a minimal standalone .tex wrapper that \\input{}s all 4 generated
tables, for a real pdflatex compile test -- not just a syntax eyeball."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "scripts" / "compile_test_wrapper.tex"

TABLES = [
    "TRACKB_MECHANISM_TABLE.tex",
    "TRACKA_MECHANISM_TABLE.tex",
    "SUPPORT_LIMITS_TABLE.tex",
    "TRACKB_MECHANISM_GAIN_TABLE.tex",
]

body = r"""\documentclass{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{pifont}
\newcommand{\wellsup}{$\bullet$}
\newcommand{\limited}{$\circ$}
\newcommand{\exploratory}{$\times$}
\begin{document}
""" + "\n".join(f"\\input{{../{t}}}" for t in TABLES) + r"""
\end{document}
"""
OUT.write_text(body)
print(f"wrote {OUT}")
