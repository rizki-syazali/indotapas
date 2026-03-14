# IndoHiTab Dataset

**IndoHiTab** is a newly constructed, high-quality dataset specifically designed to solve the Table Question Answering (TQA) task for the Indonesian language. Due to the lack of publicly available resources in this domain, this dataset serves as the primary benchmark for evaluating our proposed IndoTaPas model.

## Overview

The dataset consists of tables, natural language questions, and the corresponding answers derived from the tables. To facilitate diverse experimental setups and analyses, we provide three distinct dataset variations:

1. **IndoHiTab (Main Data):** The primary dataset manually translated from a subset of the English HiTab dataset by human annotators. After filtering out questions requiring header selection (due to TaPas architectural constraints), the dataset contains 2,559 instances, split into training and testing sets (80:20).
2. **IndoHiTab-MT (Analysis Data):** A version of the IndoHiTab training set that is automatically translated into Indonesian using a Machine Translation (MT) system. It is used to analyze system degradation when trained on MT-generated data.
3. **IndoHiTab-EXT-MT (Augmentation Data):** Extra training data created by machine-translating the remaining HiTab dataset (excluding instances already in IndoHiTab). This dataset contains 2,914 instances and is used to augment the training data during our two-stage fine-tuning process.

### Dataset Statistics

| Dataset | Train | Test | Translation Method | Purpose |
| :--- | :---: | :---: | :--- | :--- |
| **IndoHiTab** | 2,057 | 502 | Manual (Human) | Main training and evaluation |
| **IndoHiTab-MT** | 2,057 | - | Machine Translation | Analysis (Performance degradation) |
| **IndoHiTab-EXT-MT** | 2,914 | - | Machine Translation | Augmentation (Two-stage fine-tuning) |

## Dataset Variations & Formats

To ensure reproducibility and accommodate diverse research needs, we provide the IndoHiTab dataset in three structural variations: **Flattened**, **Unflattened**, and the **Original Annotated Data**.

### 1. Flattened Format
In this format, Tables with multi-level (hierarchical) column headers are preprocessed and converted into a flat, single-level header structure. This is achieved by concatenating the top-level header, which generally represents a broader category, with the lower-level headers that provide more specific subcategories. The resulting table includes a dedicated `header` array, and string values are generally lowercased.


**Example:**
```json
{
  "id": "00d47c9aac5050539645dcae34f78570",
  "question_type": ["none"],
  "question": "berapa persen penduduk canada dengan pendapatan rumah tangga sebesar $150,000 atau lebih yang mengatakan bahwa pengaruh hoki terhadap identitas nasional sangat penting?",
  "answer_coordinates": [[21, 5]],
  "answer_text": ["49"],
  "table_source": "statcan",
  "table": {
    "data": [
      ["total", "70", "69", "64", "55", "46"],
      ["kelompok umur", "", "", "", "", ""],
      ["kelompok umur 15-24", "70", "56", "51", "48", "47"],
      ["kelompok umur 25-34", "71", "65", "59", "51", "50"],
      ["kelompok umur 35-44", "75", "71", "66", "57", "48"],
      ["kelompok umur 45-54", "71", "73", "66", "57", "46"],
      ["kelompok umur 55-64", "68", "71", "65", "56", "43"],
      ["kelompok umur 65-74", "67", "74", "72", "57", "42"],
      ["kelompok umur 75", "63", "78", "77", "63", "46"],
      ["tingkat pendidikan tertinggi yang ditamatkan", "", "", "", "", ""],
       ["Tingkat pendidikan tertinggi yang ditamatkan -> Di bawah sekolah menengah","65","68", "66", "60", "46"],
      ["Tingkat pendidikan tertinggi yang ditamatkan -> Sekolah menengah", "71","69","64","58","49"],
      ["..."],
      ["identitas aborigin", "", "", "", "", ""],
      ["identitas aborigin", "74", "72", "68", "62", "54"],
      ["identitas aborigin bukan", "70", "69", "64", "54", "46"]
    ],
    "header": [
      "karakteristik sosio-demografi dan ekonomi",
      "piagam hak dan kebebasan persen",
      "bendera persen",
      "lagu kebangsaan persen",
      "rcmp persen",
      "hoki persen"
    ],
    "id": "2613",
    "title": "persepsi simbol nasional sebagai hal yang sangat penting untuk identitas kanada, berdasarkan karakteristik sosio-demografi dan ekonomi, 2013"
  }
}
```
### 2. Unflattened Format
In this format, both the headers and the cell contents are combined directly within the `data` matrix rather than being separated. The tables in this dataset consist of both **column headers** and **row headers**. The original hierarchical structure of the **row headers** is retained and represented inline using the `->` symbol. 

> **Note:** While we provide this unflattened version for completeness and potential future research, it was **not utilized** in the current study for evaluating the IndoTaPas model.

**Example:**
```json
{
  "id": "00d47c9aac5050539645dcae34f78570",
  "question_type": ["none"],
  "question": "Berapa persen penduduk Canada dengan pendapatan rumah tangga sebesar $150,000 atau lebih yang mengatakan bahwa pengaruh hoki terhadap identitas nasional sangat penting?",
  "answer_coordinates": [[23, 5]],
  "answer_text": ["49"],
  "table_id": "2613",
  "table_source": "statcan",
  "table": {
    "data": [
      [
        "Karakteristik sosio-demografi dan ekonomi",
        "Piagam hak dan kebebasan",
        "Bendera",
        "Lagu kebangsaan",
        "Rcmp",
        "Hoki"
      ],
      ["Total", "70", "69", "64", "55", "46"],
      ["Kelompok umur", "", "", "", "", ""],
      ["Kelompok umur -> 15-24", "70", "56", "51", "48", "47"],
      ["Kelompok umur -> 25-34", "71", "65", "59", "51", "50"],
      ["Kelompok umur -> 35-44", "75", "71", "66", "57", "48"],
      ["Kelompok umur -> 45-54", "71", "73", "66", "57", "46"],
      ["Kelompok umur -> 55-64", "68", "71", "65", "56", "43"],
      ["Kelompok umur -> 65-74", "67", "74", "72", "57", "42"],
      ["Kelompok umur -> 75", "63", "78", "77", "63", "46"],
      ["Tingkat pendidikan tertinggi yang ditamatkan", "", "", "", "", ""],
      ["Tingkat pendidikan tertinggi yang ditamatkan -> Di bawah sekolah menengah","65","68", "66", "60", "46"],
      ["Tingkat pendidikan tertinggi yang ditamatkan -> Sekolah menengah", "71","69","64","58","49"],
      ["..."],
      ["Identitas aborigin", "", "", "", "", ""],
      ["Identitas aborigin -> Aborigin", "74", "72", "68", "62", "54"],
      ["Identitas aborigin -> Bukan aborigin", "70", "69", "64", "54", "46"]
    ]
  }
}
```

### 3. Original Annotated Data
In addition to the flattened and unflattened formats, we also provide the original, raw annotated dataset exactly as it was produced during the human translation process. To maintain its relational structure, this raw data is separated into two files: `questions.json` and `tables.json`.

* **`questions.json`**: Contains all the manually translated questions, the corresponding answer texts, the exact answer coordinates, and the `table_id` that links to the source table. It also includes comprehensive metadata regarding the annotation and validation process.
* **`tables.json`**: Contains the complete structural data, content, and metadata for all the tables referenced by the questions.
