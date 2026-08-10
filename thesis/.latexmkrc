### latexmk configuration for the thesis ###
#
# Build with:        latexmk
# Continuous build:  latexmk -pvc
# Clean aux files:   latexmk -c
# Wipe everything:   latexmk -C
#
# The document uses fontspec (Times New Roman) and biblatex (gost-numeric)
# with biber, so we must run xelatex + biber (NOT pdflatex/bibtex).

# 5 == use xelatex via the $xelatex command below.
$pdf_mode = 5;

# Force xelatex with sane error reporting and SyncTeX for editor jumping.
$xelatex = 'xelatex -interaction=nonstopmode -halt-on-error '
         . '-synctex=1 -file-line-error %O %S';

# biblatex backend.
$bibtex_use = 2;          # always run biber if .bcf exists
$biber       = 'biber --validate-datamodel %O %S';

# Keep all auxiliary files in build/, only the .pdf is shipped.
$out_dir = 'build';
$aux_dir = 'build';

# Default master file.
@default_files = ('main.tex');

# Files latexmk -c / -C should remove (extends the built-in list with
# biblatex / glossary / synctex / fontspec leftovers).
$clean_ext = 'synctex.gz acn acr alg bbl bcf blg brf fdb_latexmk fls '
           . 'glg glo gls idx ilg ind ist lof log lot nav out run.xml '
           . 'snm toc vrb xdv';

# Open the produced PDF in the default viewer when -pv / -pvc is used.
# Comment out on headless servers / CI.
$pdf_previewer = 'xdg-open';

# Make latexmk a bit more chatty when something goes wrong.
$silent = 0;
$warnings_as_errors = 0;
