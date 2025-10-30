PROJECT TITLE: AI Skill Gap Analyzer
DEVELOPED USING: Python, Streamlit, spaCy, Sentence-BERT (all-MiniLM-L6-v2), Plotly

PROJECT OVERVIEW

AI Skill Gap Analyzer is an intelligent system designed to analyze and compare resumes and job descriptions to identify skill matches and gaps.
It uses Natural Language Processing (NLP) techniques and Machine Learning embeddings to measure how closely a candidate’s skills align with job requirements.

MAIN FEATURES

Upload resumes and job descriptions in PDF, DOCX, or TXT format.

Extract and clean text automatically for NLP processing.

Identify technical and soft skills using spaCy and custom NER models.

Perform skill comparison using cosine similarity and Sentence-BERT model (all-MiniLM-L6-v2).

Display visual analytics such as radar chart, bar chart, heatmap, and word cloud.

Generate reports and export results as PDF or CSV.

SYSTEM MODULES

Module 1: Data Ingestion and Parsing
Allows uploading of multiple document formats (PDF, DOCX, TXT). Extracts and cleans text content for further analysis.

Module 2: Skill Extraction using NLP
Uses spaCy and BERT-based pipelines to extract both technical and soft skills from resumes and job descriptions.

Module 3: Skill Gap Analysis and Similarity Matching
Encodes extracted skills into Sentence-BERT embeddings and calculates cosine similarity to identify missing or partially matched skills.

Module 4: Visualization and Dashboard
Provides an interactive Streamlit dashboard that includes radar charts, horizontal bar charts, heatmaps, and skill word clouds for visual analysis.

TECHNOLOGIES USED

Language: Python

Frameworks: Streamlit, spaCy, Sentence-BERT

Visualization: Plotly, Matplotlib, WordCloud

Libraries: NumPy, Pandas, PyPDF2, docx

Similarity Model: all-MiniLM-L6-v2

REQUIREMENTS

Before running the project, ensure the following Python libraries are installed.
(Use the requirements.txt file provided.)

Main dependencies include:
streamlit
PyPDF2
python-docx
pandas
spacy
sentence-transformers
plotly
matplotlib
wordcloud
numpy
kaleido

HOW TO RUN IN VS CODE

Open the project folder in VS Code.

Open a new terminal window.

(Optional) Create and activate a virtual environment:

python -m venv venv

venv\Scripts\activate (for Windows)

Install dependencies using:

pip install -r requirements.txt

Download the spaCy model using:

python -m spacy download en_core_web_sm

Run the project with the following command:

streamlit run pipeline.py

FILE STRUCTURE

AI-Skill-Gap-Analyzer
│
├── pipeline.py → Main project pipeline
├── requirements.txt → List of dependencies
├── README.txt → Project documentation

CONCLUSION

The AI Skill Gap Analyzer automates the process of comparing resumes and job descriptions, helping users identify missing skills efficiently.
It bridges the gap between candidates and employers by providing a visual, data-driven approach to skill evaluation.