# Convenience targets so you don't have to remember latexindent flag spellings.
#
#   make            -> build PDF (xelatex + biber via latexmk; output in build/)
#   make fmt        -> autoformat main.tex (line wrap @ 100 cols), no stdout flood
#   make watch      -> continuous build, rebuild on save
#   make clean      -> remove build/ and latexindent backup/log droppings
#   make distclean  -> 'clean' + delete the produced PDF
#
# All settings live in .latexmkrc and .latexindent.yaml; this file is just glue.

TEX     := main.tex
BUILD   := build
PDF     := $(BUILD)/main.pdf

# `latexindent` rule:
#   * flags are passed SEPARATELY (Perl Getopt::Long doesn't bundle short flags)
#   * -s silences the stdout dump (otherwise the whole formatted file is echoed)
#   * -l auto-discovers ./.latexindent.yaml
#   * -m enables modifyLineBreaks (required for the 100-col wrap)
#   * we redirect any remaining noise to /dev/null and clean .bak* afterwards
LATEXINDENT_FLAGS := -w -l -m -s

.PHONY: all pdf fmt watch clean distclean check

all: pdf

pdf $(PDF): $(TEX)
	latexmk

fmt:
	@echo ">> formatting $(TEX) (wrap=100, in place)"
	@latexindent $(LATEXINDENT_FLAGS) $(TEX) >/dev/null
	@rm -f main.bak[0-9]* indent.log
	@echo ">> done. $(TEX) is formatted; backups removed."

# Useful in CI: returns non-zero if the file would change after formatting.
# (We copy to a *.tex file so latexindent applies the same processing pipeline.)
check:
	@cp $(TEX) main.fmt-check.tex
	@latexindent $(LATEXINDENT_FLAGS) main.fmt-check.tex >/dev/null
	@rm -f main.fmt-check.bak[0-9]* indent.log
	@if ! diff -q $(TEX) main.fmt-check.tex >/dev/null; then \
		echo "FAIL: $(TEX) is not formatted (run 'make fmt')"; \
		diff -u $(TEX) main.fmt-check.tex | head -40; \
		rm -f main.fmt-check.tex; exit 1; \
	fi
	@rm -f main.fmt-check.tex
	@echo "OK: $(TEX) is properly formatted."

watch:
	latexmk -pvc

clean:
	latexmk -c
	rm -rf $(BUILD)
	rm -f main.bak[0-9]* indent.log

distclean: clean
	rm -f $(PDF)
