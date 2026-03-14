# IndoWikiTableText Dataset

**IndoWikiTableText** is a large-scale dataset specifically constructed to facilitate the pre-training of tabular language models (such as IndoTaPas) for the Indonesian language.

## Overview

This dataset comprises **1,636,656 text-table pairs** compiled from **1,319,462 Indonesian Wikipedia pages**. It pairs structured tabular data with relevant surrounding text, enabling models to learn meaningful text-table associations and representations in Indonesian.

### Dataset Statistics

The dataset extracts two distinct types of tables from Wikipedia: **Infoboxes** and **Wikitables**.

| Source | Infobox | Wikitable | Total Text-Table Pairs | 
| :--- | :---: | :---: | :---: | 
| **1,319,462 Wikipedia Pages** | 1,047,233 | 589,423 | **1,636,656** | 

## Dataset Construction

The construction of IndoWikiTableText involves several rigorous steps, from crawling to structuring:

### 1. Data Collection

We generated a complete list of Indonesian Wikipedia pages (via the `Istimewa:Daftar_halaman` directory) and crawled over one million pages locally.

### 2. Table Extraction & Standardization

Tables were identified using the `class` attribute within the HTML `<table>` tags. We extracted two primary types:

* **Infobox:** Horizontal tables typically appearing at the top-right corner of a Wikipedia article serving as concise summaries. *Transformation:* We applied a transposition process to convert these into vertical tables with a single header and one row of data to ensure structural consistency across the dataset.
* **Wikitable:** Standard vertical tables embedded within specific subsections of the article body.

### 3. Text-Table Pairing

To allow the model to learn context, tables were paired with relevant descriptive texts sourced from their immediate surroundings in the article. These textual contexts include:

* Table titles and captions.
* Introductory or concluding paragraphs.
* The article’s main title or relevant subheadings.

## Data Format & Example

Each instance in the dataset is structured as a JSON object containing the Wikipedia page metadata, the extracted table, and the surrounding contextual text. 

Here is an example of a single text-table pair (an Infobox) from the dataset:

```json
{
  "page_id": 1422142,
  "url": "[https://id.wikipedia.org/wiki?curid=1422142](https://id.wikipedia.org/wiki?curid=1422142)",
  "table": {
    "data": [
      [
        "abad ke-3, Aleksandria, Mesir",
        "25 November 311, Aleksandria, Mesir",
        "Ortodoks Oriental, Ortodoks Timur, Katolik Roma",
        "25 November"
      ]
    ],
    "header": [
      "Lahir",
      "Meninggal",
      "Dihormati di",
      "Pesta"
    ],
    "type": "infobox"
  },
  "text": "Paus Petrus dari Aleksandria adalah Paus Aleksandria ke-17 & Patriarkh Tahta St. Markus. Ia diakui sebagai santo oleh Gereja Ortodoks Koptik, Gereja Katolik Roma, dan Gereja Ortodoks Timur."
}