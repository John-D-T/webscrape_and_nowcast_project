import subprocess
from texttable import Texttable
import latextable


def save_df_as_image(df, file_name):
    # https://economics.stackexchange.com/questions/11774/outputting-regressions-as-table-in-python-similar-to-outreg-in-stata
    filename = '%s.tex' % file_name
    pdffile = '%s.pdf' % file_name
    outname = '%s.png' % file_name

    template = r'''\documentclass[preview]{{standalone}}
    \usepackage{{booktabs}}
    \begin{{document}}
    {}
    \end{{document}}
    '''

    with open(filename, 'w+') as f:
        f.write(template.format(df.to_latex()))

    subprocess.call(['pdflatex', filename])
    subprocess.call(['convert', '-density', '300', pdffile, '-quality', '90', outname])

def save_table_as_latex(file_name, rows, header_count, caption):

    table = Texttable()
    table.set_cols_align(["c"] * header_count)
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows(rows)

    table = latextable.draw_latex(table, caption=caption)

    filename = '%s.tex' % file_name
    pdffile = '%s.pdf' % file_name
    outname = '%s.png' % file_name

    beginningtex = """\\documentclass{report}
    \\usepackage{booktabs}
    \\usepackage{longtable}
    \\usepackage{booktabs}
    \\usepackage{makecell}
    \\usepackage[landscape,hmargin=0.5cm]{geometry}
    \\begin{document}
    \\begin{flushleft}"""
    endtex = """\\end{flushleft}
    \\end{document}"""

    with open(filename, 'w+') as f:
        f.write(beginningtex)
        # f.write(model.summary().as_latex())
        f.write(table)
        f.write(endtex)

    subprocess.call(['pdflatex', filename])
    subprocess.call(['convert', '-density', '300', pdffile, '-quality', '90', outname])

def save_model_as_image(model, file_name, lin_reg=False):
    # https://economics.stackexchange.com/questions/11774/outputting-regressions-as-table-in-python-similar-to-outreg-in-stata
    filename = '%s.tex' % file_name
    pdffile = '%s.pdf' % file_name
    outname = '%s.png' % file_name

    # template = r'''\documentclass{report}
    # \usepackage{{booktabs}}
    # \begin{{document}}
    # {}
    # \end{{document}}
    # '''
    beginningtex = """\\documentclass{report}
    \\usepackage{booktabs}
    \\usepackage{longtable}
    \\usepackage{booktabs}
    \\usepackage{makecell}
    \\usepackage[landscape,hmargin=0.5cm]{geometry}
    \\begin{document}
    \\begin{flushleft}"""
    endtex = """\\end{flushleft}
    \\end{document}"""

    with open(filename, 'w+') as f:
        f.write(beginningtex)
        if lin_reg:
            f.write(model.summary.as_latex())
        else:
            f.write(model.summary().as_latex())
        f.write(endtex)

    subprocess.call(['pdflatex', filename])
    subprocess.call(['convert', '-density', '300', pdffile, '-quality', '90', outname])