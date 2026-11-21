# Normalizing Flows project
#   make setup        create a Python virtual environment and install packages
#   make check        verify every flow layer (invertibility + log-determinant)
#   make quick        run all experiments briefly (~2 min) to test the pipeline
#   make experiments  run all experiments in full (~20-45 min)
#   make              build main.pdf from whatever figures/tables exist
#   make everything   experiments + PDF
#   make watch        rebuild the PDF on every save
#   make clean/purge  remove LaTeX build files (purge also removes the PDF)

MAIN   = main
VENV   = .venv
PYTHON = $(VENV)/bin/python

.PHONY: all setup check quick experiments everything watch clean purge

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex bibliography.bib $(wildcard figures/*.pdf) $(wildcard results/*.tex)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

setup:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r code/requirements.txt

check:
	cd code && ../$(PYTHON) 00_check_flows.py

quick:
	cd code && PATH="$(CURDIR)/$(VENV)/bin:$$PATH" bash run_all.sh --quick

experiments:
	cd code && PATH="$(CURDIR)/$(VENV)/bin:$$PATH" bash run_all.sh

everything: experiments all

watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(MAIN).tex

clean:
	latexmk -c
	rm -f *.bbl *.run.xml

purge:
	latexmk -C
	rm -f *.bbl *.run.xml
