import time

import cv2
import tqdm
from doctr.io import DocumentFile
import numpy as np
from fuzzywuzzy import fuzz
from doctr.models import ocr_predictor
import json
import pandas as pd
import sys


if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


class ParseReport:
    """
    Attributes
    ----------
    words : pd.Dataframe
        Contains each word of the documents separately with unique identifier. The data type is pd.Dataframe.
        Unique Identifiers: page_idx, block_idx, line_idx, and word_idx.

    Methods
    -------
    find_attribute_one(self, word, extract_value=True, context=None)
        Finds attributes values with one word.

    find_attribute_two(self, word1, word2, extract_value=True, context=None)
        Finds attributes values with two words.

    company_name(self)
        First line of the document, and assign as a company name.

    extract_value(sentence):
        Separates a string with key value combination, and returns value.

    get_lines(self, word1, word2, context=None):
        Finds lines containing two words from the whole document.

    get_lien(self):
        Finds information about 'Lien Types'.

    get_vesting_instrument(self):
        Finds information about 'Vesting Instrument Type'

    get_instrument(self):
        Finds information about 'Instrument Type'.

    find_table_pages(self, word1, word2):
        Finds pages that can have 'Federal Tax Lien' tables.

    find_column_values(context, word1, word2, right=0.0, left=0.0, height=0.21):
        Finds values for a specific column for a specific table.

    get_lien_tables(self):
        Finds all the value in a structure way for a table.
    """
    def __init__(self, file, json_file=False, save_json=False):
        if json_file:
            df = pd.read_json(file)
        else:
            model = ocr_predictor(pretrained=True)
            doc = DocumentFile.from_images(file)
            result = model(doc)
            json_output = result.export()
            df = pd.DataFrame(json_output)

            if save_json:
                with open(save_json, "w") as f:
                    json.dump(json_output, f, indent=2)

        pages = df.join(pd.json_normalize(df.pop('pages')))

        blocks = pages.explode("blocks")
        blocks['block_idx'] = np.arange(blocks.shape[0])
        blocks['index'] = blocks['block_idx']
        blocks = blocks.set_index('index')

        blocks = blocks.join(pd.json_normalize(blocks.pop('blocks')))
        blocks = blocks.rename(columns={'geometry': 'block_geometry'})

        self.lines = blocks.explode("lines")
        self.lines['line_idx'] = np.arange(self.lines.shape[0])
        self.lines['index'] = np.arange(self.lines.shape[0])
        self.lines = self.lines.set_index('index')

        self.lines = self.lines.join(pd.json_normalize(self.lines.pop('lines')))
        self.lines = self.lines.rename(columns={'geometry': 'line_geometry'})

        words = self.lines.explode("words")
        words['word_idx'] = np.arange(words.shape[0])
        words['index'] = np.arange(words.shape[0])
        words = words.set_index('index')

        words = words.join(pd.json_normalize(words.pop('words')))
        words = words.rename(columns={'geometry': 'word_geometry'})

        words["word_geometry"] = words.word_geometry.apply(lambda x: {"x1": x[0][0], "y1": x[0][1], "x2": x[1][0], "y2": x[1][1]})

        self.words = words.join(pd.json_normalize(words.pop('word_geometry')))

    def company_name(self):
        """
        First line of the document, and assign as a company name.

        :return: string (company name)
        """
        return " ".join(self.words[(self.words['page_idx'] == 0) & (self.words['block_idx'] == 0) & (self.words['line_idx'] == 0)]["value"].values)


if __name__ == '__main__':
    import argparse
    time.sleep(20)
    parser = argparse.ArgumentParser(
        prog='ParseReport',
        description='It parses report',
        epilog='Thanks')

    parser.add_argument('-f', '--file', required=True)
    parser.add_argument('-j', '--json', required=False, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-a', '--save_json_path', required=False, default=False)
    args = parser.parse_args()
    print("Extracting information for file:", args.file)

    report = ParseReport(file=args.file, json_file=args.json, save_json=args.save_json_path)
    print(report.company_name())
    # print(report.words.to_string())
    print(report.words.columns)
    line = 0
    height = 900
    width = 1000
    frame = np.ones((height, width))
    progress = tqdm.tqdm(total=report.words.shape[0])
    for i, word in report.words.iterrows():
        # print(word["line_idx"], word["value"])
        x, y = int(width * (word["x1"])), int(height * word["y2"]) + 20
        print(x, y, word["value"])
        cv2.putText(frame, f'{word["value"]}', (int(x), int(y)), cv2.QT_FONT_NORMAL, 0.5, (0, 222, 222), 1)
        cv2.imshow('recreate', frame)
        time.sleep(0.3)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        progress.update(1)
