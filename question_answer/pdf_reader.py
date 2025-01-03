import re
import fitz
import pandas as pd


def cleaner(text):
    """
        Cleans dirty text.
        Args:
            - text to clean.
        Returns:
            - cleaned text.
    """
    rules = [
        r"Page \d+( of \d+)?",  # remove Page annotations
        r"Pages \d+(-\d+)?",  # remove Page annotations
        r"\d{2}-\d{2}-\d{2}",  # remove chapter/subchapter numbers
        r"[^\n]*\nFigure \d+ \(Sheet \d+\)",  # remove Figure annotations
        r"ﬁ", "fi",
        r"ﬂ", "fl",
        r"© Cessna Aircraft Company",
        r"CONTENTS",
        r"Jun 1/2005",
        r"CESSNA AIRCRAFT COMPANY",
        r"SINGLE ENGINE",
        r"STRUCTURAL REPAIR MANUAL",
        r"DAMAGE INVESTIGATION AND CLASSIFICATION",
        r"^\s*$\n"
    ]

    for rule in rules:
        text = re.sub(rule, "", text, flags=re.MULTILINE)

    return text


def splitter(text_to_slit):
    """
        Splits long texts into shorter ones section by section based by a predefined rules.
        Rule:
            - splitting long texts into sections using "1." number format as a divider.
        Args:
            - text to split.
        Returns:
            - split texts.
    """
    # Regular expression to split by sections starting with numbers followed by "."
    sections = re.split(r'(?<=\n)(?=\d+\.\n)', text_to_slit)

    # Removing leading/trailing spaces for clarity in each section
    sections = [section.strip() for section in sections if section.strip()]

    return sections


def read_doc(file):
    """
        Reads, cleans, splits a doc.
        Args:
            - file to read.
        Returns:
            - a df with chapters, titles and sub-docs.
    """
    doc = fitz.open(file)

    # Use table of contents (toc) as a reference point
    toc = doc.get_toc()
    needed_toc = []
    not_needed_toc = [
        'GENERAL',
        'TABLE OF CONTENTS',
        'COVERAGE',
        'AIRPLANE IDENTIFICATION',
        'AEROFICHE (MICROFICHE)',
        'USING THE STRUCTURAL REPAIR MANUAL OR AEROFICHE',
        'REVISION (MANUAL)',
        'TABLE OF CONTENTS',
        'IDENTIFYING REVISED MATERIAL',
        'LIST OF EFFECTIVE PAGES',
        'RECORD OF TEMPORARY REVISIONS'
    ]

    # Populate a list with needed titles
    for i in toc:
        if (i[0] == 2 or i[0] == 3) and i[1] not in not_needed_toc:
            needed_toc.append(i)

    # Read the document title by title
    final = []
    chapter = None
    for i, entry in enumerate(needed_toc):
        text = ""
        level, title, start_page = entry
        if entry[0] != 2:
            end_page = needed_toc[i + 1][2] if i + 1 < len(needed_toc) else len(doc)

            # Extract text from the chapter's page range
            for page_num in range(start_page - 1, end_page - 1):
                page = doc[page_num]
                text += page.get_text()

            if len(text) > 2000:
                text = splitter(text)
            else:
                text = [text]

            for j in text:
                final.append({'chapter': chapter, 'title': title, 'text': cleaner(j)})
        else:
            chapter = title

    df = pd.DataFrame(final)
    df.to_csv('question_answer/split_srm.csv', index=False)

    return df
