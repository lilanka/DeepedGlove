#!/bin/sh

pdflatex doc.tex
rm -r *.aux *.log *.toc *.out
xdg-open doc.pdf
