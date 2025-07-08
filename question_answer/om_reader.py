import pandas as pd
import fitz
import re


def read_overhaul_manual(pdf_file):
    doc = fitz.open(pdf_file)
    toc = doc.get_toc()
    needed_toc = [
        ['Hose & Tubine Installation', 34],
        ['Engine Removal Instructions', 39],
        ['Preliminary Cleaning', 40],
        ['General Cleaning', 42],
        ['Visual Inspection', 43],
        ['Ultrasonic Inspection', 44],
        ['Helical Coil Insert Replacement', 46],
        ['Fuel Nozzles', 99],
        ['Cleaning', 100],
        ['Starter Motor Overhaul', 139],
        ['Starter Adapter Assembly', 152],
        ['Cylinder Disassembly', 186],
        ['Exhaust', 314],
        ['Fuel Injection', 316],
        ['Engine Test', 332],
        ['Fuel System Adjustment', 333],
    ]

    final_text = ''

    for _, title, page_num in toc:
        pair_to_find = [title, page_num]
        found = any(item == pair_to_find for item in needed_toc)
        if found:
            # PDF page numbers are zero-based in PyMuPDF
            page = doc[page_num - 1]
            text = page.get_text()
            cleaned_text = re.sub(r'^\s*\d+-\d+[A-Z]*\s+', '', text, flags=re.MULTILINE)
            cleaned_text = re.sub(r'\(See Figure \d+-\d+[A-Z]*\)', '', cleaned_text)
            cleaned_text = re.sub(r'^\s*[A-Z]+\s+\d{4}\s*$', '', cleaned_text, flags=re.MULTILINE)
            cleaned_text = re.sub(r'^\s*FIGURE\s+\d+-\d+\.\s+.*$', '', cleaned_text, flags=re.MULTILINE)
            cleaned_text = re.sub(r'^\s*FIGURE\s+\d+-\d+[A-Z]*\.\s*.*$', '', cleaned_text, flags=re.MULTILINE)
            cleaned_text = re.sub(r'\n+', '\n', cleaned_text).strip()

            final_text += cleaned_text + '\n'
    return final_text


def logical_split(text, max_length=2000):
    # Find all positions where a numbered step starts (e.g., '1.', '2.', ... at start of line)
    step_pattern = re.compile(r'^(\d+\.)', re.MULTILINE)
    steps = list(step_pattern.finditer(text))
    chunks = []

    if not steps or len(text) <= max_length:
        return [text.strip()]

    # Split into logical steps
    split_points = [match.start() for match in steps] + [len(text)]

    # Group steps into chunks <= max_length
    i = 0
    while i < len(split_points) - 1:
        chunk_start = split_points[i]
        # Try to fit as many steps as possible under max_length
        for j in range(i + 1, len(split_points)):
            candidate = text[chunk_start:split_points[j]].strip()
            if len(candidate) > max_length:
                # Too long, take up to previous
                if j == i + 1:  # Even one step is too big, force split
                    chunks.append(candidate)
                    i = j
                else:
                    last_good = text[chunk_start:split_points[j - 1]].strip()
                    chunks.append(last_good)
                    i = j - 1
                break
        else:
            # All remaining steps fit in one chunk
            chunks.append(text[chunk_start:split_points[-1]].strip())
            break

    return chunks


def prepare_overhaul_manual(overhaul_manual='question_answer/om.pdf', cleaner=lambda x: x):
    text = read_overhaul_manual(overhaul_manual)
    # Pattern matches uppercase lines, but NOT "WARNING" (using negative lookahead)
    title_pattern = re.compile(r'^(?!WARNING$)([A-Z][A-Z\s]+)$', re.MULTILINE)
    final = []
    matches = list(title_pattern.finditer(text))

    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        cleaned_body = cleaner(body)
        split_bodies = logical_split(cleaned_body)
        for split_text in split_bodies:
            final.append({'chapter': None, 'title': title, 'text': split_text})
    df = pd.DataFrame(final)
    df.to_csv('question_answer/split_om.csv', index=False)

    return df
