# A REPORT

## ON

## INTELLIGENT QUESTION ANSWERING SYSTEM USING CLASSICAL NLP AND SEMANTIC SEARCH

### By

**Name of the Student:** `<Your Full Name>`  
**Registration Number:** `<Your Registration Number>`

Prepared in the partial fulfillment of the project component of  
**Course CSE 423 - Natural Language Processing**

**Department of Computer Science and Engineering**  
**SRM University, AP**

**APRIL 2026**

---

# ACKNOWLEDGEMENTS

I express my sincere gratitude to the Vice Chancellor and the Dean of SRM University, AP, for providing the academic environment and infrastructure required for the completion of this project. I extend my heartfelt thanks to the faculty of the Department of Computer Science and Engineering for encouraging project-based learning and practical exploration in the area of Natural Language Processing.

I am especially grateful to **`<Faculty Mentor Name>`**, Faculty Mentor, Department of Computer Science and Engineering, for the valuable guidance, continuous encouragement, and constructive suggestions provided throughout the development of this work. Their feedback helped shape the direction, implementation, and presentation of the project.

I also thank **`<Course Instructor Name>`** and all other faculty members who supported me during the course of this work. I would like to acknowledge my classmates, friends, and family members for their constant motivation and support during the preparation of this project and report.

Finally, I acknowledge the developers and research communities behind open-source tools such as Python, spaCy, Sentence-Transformers, FAISS, BM25, Plotly, and Streamlit, which made the implementation of this system possible.

---

# CERTIFICATE

This is to certify that the project entitled **"Intelligent Question Answering System Using Classical NLP and Semantic Search"** has been successfully completed by **`<Student Full Name>`** (Roll Number: **`<Student Roll/ID Number>`**) as a required academic component of the course **CSE 423: Natural Language Processing** during the Academic Year **2025-26** in the Department of Computer Science and Engineering at **SRM University, AP**. This work was carried out under the guidance of the undersigned and is deemed to have satisfactorily fulfilled the course requirements as of **`<Date>`**.

  
  
**(Signature of the Course Instructor / Faculty Mentor)**  
**`<Instructor Name>`**  
**`<Designation>`**  
Department of Computer Science and Engineering  
SRM University, AP

---

# ABSTRACT

This project presents an **Intelligent Question Answering System (IQAS)** that answers user questions from uploaded documents using classical Natural Language Processing and semantic retrieval techniques. The objective of the system is to allow users to upload study materials such as PDFs, TXT files, and DOCX files, convert them into a searchable knowledge base, and obtain precise answers with source references. The system has been implemented in Python with a Streamlit-based user interface and a modular backend pipeline.

The workflow includes document loading, text cleaning, smart chunking, sentence embedding generation, vector indexing with FAISS, keyword retrieval with BM25, reciprocal rank fusion, cross-encoder re-ranking, and extractive answer generation. The system also identifies question types, highlights named entities, and provides an analytics dashboard showing confidence trends, question distribution, and document coverage.

The project demonstrates how a compact, CPU-based NLP pipeline can be used to build a practical question answering application without requiring a full generative language model. The final outcome is a working prototype that is modular, reusable, and suitable for educational document search and question answering tasks.

---

# TABLE OF CONTENTS

Update the page numbers after final formatting in Word.

| Section | Title | Page No. |
|---|---|---|
| 1 | Introduction | `To be updated` |
| 2 | Problem Statement and Objectives | `To be updated` |
| 3 | System Overview | `To be updated` |
| 4 | Main Text | `To be updated` |
| 4.1 | Assumptions Made | `To be updated` |
| 4.2 | Experimental Setup and Data Handling | `To be updated` |
| 4.3 | Methodology and Algorithms | `To be updated` |
| 4.4 | Architecture Diagram / Flow Chart | `To be updated` |
| 4.5 | Module-Wise Description | `To be updated` |
| 4.6 | Implementation Details | `To be updated` |
| 4.7 | Discussion of Results | `To be updated` |
| 5 | Outcomes | `To be updated` |
| 6 | Conclusions and Recommendations | `To be updated` |
| 7 | Code | `To be updated` |
| 8 | Appendices | `To be updated` |
| 9 | References | `To be updated` |

---

# 1. INTRODUCTION

The rapid growth of digital content has created a need for intelligent systems that can extract useful information from unstructured text. Students, teachers, researchers, and professionals often work with large collections of lecture notes, reports, books, and articles. Manually locating relevant information inside such documents is time-consuming and inefficient. A question answering system addresses this problem by allowing users to ask natural language questions and receive direct answers from the content of available documents.

This project focuses on the design and implementation of an **Intelligent Question Answering System (IQAS)** that works on uploaded documents. Instead of relying only on keyword search, the system combines multiple NLP techniques such as tokenization, part-of-speech analysis, named entity recognition, semantic embeddings, dense retrieval, sparse retrieval, and answer extraction. The goal is to improve both relevance and usability while keeping the system lightweight enough to run on a CPU-based local environment.

The project is especially relevant to the field of Natural Language Processing because it demonstrates the integration of several foundational NLP tasks into one end-to-end application. It also shows how theoretical concepts from the course can be applied to a real software system with a user interface, persistent indexing, and analytics.

---

# 2. PROBLEM STATEMENT AND OBJECTIVES

## 2.1 Problem Statement

Users frequently need to ask questions about the contents of documents such as notes, reports, and study materials. Traditional search methods require exact keywords and often return large text blocks without summarization or context. There is therefore a need for a system that can process uploaded documents, understand user questions, retrieve the most relevant information, and present a concise answer along with its source.

## 2.2 Objectives

The objectives of this project are:

- To build a document-based question answering system for PDF, TXT, and DOCX files.
- To preprocess and organize document text into retrievable units using smart chunking.
- To apply sentence embeddings and vector indexing for semantic search.
- To combine dense retrieval and sparse retrieval for better relevance.
- To generate extractive answers with source citation, question type, and confidence score.
- To provide a user-friendly interface for uploading files, asking questions, and viewing analytics.
- To create a modular software design that is easy to extend in future.

---

# 3. SYSTEM OVERVIEW

The project is implemented as a multi-module Python application with a Streamlit front end. The system starts by accepting one or more documents from the user. These documents are loaded and cleaned, then split into smaller text chunks using one of three strategies: fixed-length, sentence-aware, or paragraph-aware chunking. The chunks are then converted into dense vector embeddings using a Sentence-Transformers model.

The embeddings are indexed using FAISS to support fast semantic retrieval. In parallel, the same chunks are also searched using BM25 for keyword-based matching. The dense and sparse results are combined using Reciprocal Rank Fusion (RRF), and the top candidates are optionally re-ranked using a cross-encoder model. From the best passage, the answer extractor selects the most relevant sentences, estimates confidence, identifies entities, and returns the answer with the source filename and page number.

The user interface contains three main pages:

- Upload page for document ingestion and index building.
- Q&A page for asking questions and viewing answers.
- Analytics page for exploring question patterns, confidence trends, and entity statistics.

---

# 4. MAIN TEXT

## 4.1 Assumptions Made

The following assumptions are made in the current version of the project:

- The input documents are primarily in English.
- The text in uploaded PDF files is machine-readable and not image-only.
- The spaCy English language model is available during setup.
- The system is intended for local or small-scale use and runs on CPU.
- The answers are extractive, meaning they are formed from text present in the documents rather than generated from external knowledge.
- The user builds the index before attempting to ask questions.

## 4.2 Experimental Setup and Data Handling

The project does not depend on a fixed external benchmark dataset. Instead, it is designed to work on **user-uploaded documents**, which makes it adaptable to study materials, reports, lecture notes, or domain-specific documents. During development, the repository includes a sample text document containing NLP lecture notes, which can be used for demonstration and testing.

The data flow begins with the ingestion of supported file formats:

- PDF using PyMuPDF
- TXT using standard UTF-8 text reading
- DOCX using python-docx

Each uploaded file is stored in the data directory, processed page by page or section by section, and transformed into a structured internal representation containing:

- document identifier
- filename
- source path
- page number
- total pages
- extracted text

This design helps preserve provenance so that answers can later be traced back to the correct source document and page.

## 4.3 Methodology and Algorithms

### 4.3.1 Document Loading

The **DocumentLoader** module reads uploaded files and converts them into normalized internal objects. PDFs are processed page-wise to preserve page metadata. TXT and DOCX files are loaded as single logical sections.

### 4.3.2 Text Cleaning

The system applies text normalization to reduce noise introduced by document extraction. This includes:

- fixing hyphenated line breaks
- Unicode normalization
- removing non-printable characters
- removing form feeds and null bytes
- reducing extra whitespace
- preserving paragraph structure where useful

This stage helps improve the quality of downstream tokenization and retrieval.

### 4.3.3 Smart Chunking

The cleaned text is split into retrievable chunks using one of three strategies:

- **Fixed-size chunking**: divides text by token count with overlap.
- **Sentence-aware chunking**: groups full sentences without splitting in the middle.
- **Paragraph-aware chunking**: respects paragraph boundaries while limiting size.

Chunking is important because it determines the granularity of retrieval. If chunks are too large, retrieval becomes noisy. If chunks are too small, context may be lost.

### 4.3.4 Embedding Generation

Each chunk is transformed into a dense vector representation using the Sentence-Transformers model **`all-MiniLM-L6-v2`**. This model produces 384-dimensional embeddings and is suitable for lightweight semantic similarity tasks. The embeddings are L2-normalized and cached on disk to avoid recomputation.

### 4.3.5 FAISS Indexing

The vector embeddings are stored using **FAISS**, which enables efficient nearest-neighbor search. For smaller corpora, the system uses a flat inner-product index. For larger corpora, the design supports an IVF-based FAISS index. The index and associated metadata are saved to disk so that the application can reload them later without rebuilding.

### 4.3.6 Sparse Retrieval with BM25

In addition to semantic search, the system uses **BM25** to capture exact keyword matches. This is useful when important terms in the question must appear literally in the relevant passage.

### 4.3.7 Reciprocal Rank Fusion

Dense retrieval and sparse retrieval each produce their own ranked list of candidate chunks. The system merges them using **Reciprocal Rank Fusion (RRF)**, which rewards chunks that appear high in either ranking and especially those that appear in both.

### 4.3.8 Cross-Encoder Re-ranking

The top fused candidates are optionally re-ranked using a cross-encoder model, specifically **`ms-marco-MiniLM-L-6-v2`**. This model jointly scores the question and passage pair, improving the fine-grained ordering of results.

### 4.3.9 Answer Extraction

The **AnswerExtractor** module selects the highest-ranked passage and identifies the most relevant sentences within it. It uses question type detection and sentence-level semantic similarity to decide how much text to include in the final answer. The output includes:

- answer text
- source filename
- page number
- confidence score
- supporting passage
- extracted entities
- question type

### 4.3.10 Question Analysis

The project also performs question analysis through part-of-speech tagging and rule-based question type detection. Supported question classes include:

- WHO
- WHAT
- WHEN
- WHERE
- WHY
- HOW
- DEFINE
- OTHER

This analysis supports better answer formatting and later analytics.

## 4.4 Architecture Diagram / Flow Chart

The overall workflow of the system is shown below.

```text
                    +----------------------+
                    |   User Uploads File  |
                    |  PDF / TXT / DOCX    |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   Document Loader    |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |    Text Cleaning     |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   Smart Chunking     |
                    | Fixed / Sentence /   |
                    | Paragraph            |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Sentence Embeddings  |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |    FAISS Indexing    |
                    +----------+-----------+
                               |
       +-----------------------+------------------------+
       |                                                |
       v                                                v
+----------------------+                     +----------------------+
| Dense Semantic Search |                     | BM25 Keyword Search  |
+----------+-----------+                     +----------+-----------+
           \                                             /
            \                                           /
             v                                         v
                    +----------------------+
                    | Reciprocal Rank      |
                    | Fusion (RRF)         |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Cross-Encoder        |
                    | Re-ranking           |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Answer Extraction    |
                    | + Confidence +       |
                    | Source Citation      |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Streamlit Interface  |
                    | Q&A + Analytics      |
                    +----------------------+
```

## 4.5 Module-Wise Description

### 4.5.1 Core Pipeline

The `QAPipeline` class acts as the orchestrator of the entire system. It lazily initializes components, handles document ingestion, loads saved indexes, and answers user queries.

### 4.5.2 NLP Modules

The NLP layer is divided into:

- tokenizer
- POS tagger
- named entity recognizer
- embedder
- similarity utilities

This separation improves readability and makes the project easier to maintain and extend.

### 4.5.3 Retrieval Layer

The retrieval layer consists of:

- FAISS indexer
- BM25 retriever
- fusion and re-ranking logic

This hybrid design balances semantic similarity with keyword-level precision.

### 4.5.4 User Interface

The Streamlit front end simplifies user interaction. The upload page allows document ingestion and index building. The Q&A page handles question input, answer display, entity highlighting, supporting passage display, and retrieval score inspection. The analytics page visualizes query distribution, confidence timeline, top entities, and document coverage.

### 4.5.5 Logging and Configuration

Centralized configuration enables easy control of model names, chunk sizes, retrieval limits, paths, and UI settings. Structured logging is used for debugging and monitoring execution flow.

## 4.6 Implementation Details

The main technologies used in the project are listed below:

- **Python** for implementation
- **Streamlit** for the web-based user interface
- **spaCy** for tokenization, part-of-speech tagging, and named entity recognition
- **Sentence-Transformers** for embedding generation
- **FAISS** for vector indexing and dense retrieval
- **rank-bm25** for sparse retrieval
- **Transformers / CrossEncoder** for passage re-ranking
- **Plotly and Pandas** for analytics visualization
- **Docker and Docker Compose** for containerized deployment

The project structure is organized as follows:

- `app/` for the Streamlit interface
- `core/` for pipeline, loader, retriever, indexer, and answer extractor
- `nlp/` for tokenizer, POS tagging, NER, embeddings, and similarity
- `utils/` for configuration, cleaning, chunking, and logging
- `tests/` for pipeline and module tests
- `models/` for cached embeddings and FAISS indexes
- `data/` for uploads and sample documents

## 4.7 Discussion of Results

The implemented prototype demonstrates that a practical document question answering system can be built by combining classical NLP with semantic retrieval. The system supports multiple document formats, preserves document provenance, and produces answers with confidence and source reference rather than returning only a list of matched files.

The use of hybrid retrieval improves robustness. BM25 is useful for exact keywords such as technical terms, while dense retrieval helps when the question and the document express similar meaning using different wording. Re-ranking further improves the ordering of retrieved passages. The answer extractor then selects the most relevant sentences so that the final response is shorter and more useful than the full passage.

The system also includes quality-of-life features such as persistent indexing, chunking configuration, question type detection, named entity highlighting, and an analytics dashboard. These additions make the project more complete than a basic search demo.

At the same time, the project has some limitations. The answers are extractive and depend on the quality of the retrieved passage. The current pipeline is primarily designed for English text. The analytics are session-oriented rather than fully persistent. Also, some software engineering aspects such as portability of logging behavior and more comprehensive benchmark evaluation could be improved in future versions.

---

# 5. OUTCOMES

The principal outcomes of the project are:

- A working **document-based question answering application** with a graphical user interface.
- Support for **PDF, TXT, and DOCX** document ingestion.
- Implementation of **text cleaning** and **smart chunking** strategies.
- Use of **Sentence-Transformers** embeddings for semantic search.
- Use of **FAISS** for persistent dense vector indexing.
- Use of **BM25** for sparse keyword-based retrieval.
- Use of **Reciprocal Rank Fusion** to combine dense and sparse search results.
- Use of **cross-encoder re-ranking** for improved relevance of top passages.
- Generation of **extractive answers with source citation, page number, and confidence score**.
- Addition of **question type detection**, **named entity extraction**, and **entity highlighting**.
- Addition of an **analytics dashboard** for question trends, confidence, and document coverage.
- A modular project structure suitable for future enhancement and deployment.

---

# 6. CONCLUSIONS AND RECOMMENDATIONS

## 6.1 Conclusion

This project successfully demonstrates the design and implementation of an Intelligent Question Answering System using classical NLP and semantic retrieval techniques. The system solves a meaningful problem by allowing users to ask natural language questions over uploaded documents and receive concise, source-aware answers. The integration of document ingestion, preprocessing, chunking, embeddings, FAISS indexing, BM25 retrieval, fusion, re-ranking, answer extraction, and analytics makes the project comprehensive and academically relevant to the field of Natural Language Processing.

The final prototype shows that even without a full generative large language model, an effective and practical QA system can be built using lightweight and well-structured components. The project therefore serves as a strong example of applied NLP, information retrieval, and software integration.

## 6.2 Recommendations and Future Work

The following improvements are recommended for future versions:

- Add **generative answer synthesis** using an LLM for more natural responses.
- Add **multilingual support** for non-English documents.
- Support **OCR for scanned PDFs** containing image-based text.
- Add **user authentication and document management** for multi-user deployment.
- Improve **evaluation metrics** using benchmark datasets such as Exact Match, F1 score, precision, and recall.
- Provide a **REST API** for integration with other applications.
- Improve analytics persistence by storing query logs in a database.
- Optimize for **GPU acceleration** for faster embedding and re-ranking.
- Improve engineering robustness in logging and testing across different environments.

---

# 7. CODE

The full code of the project should be attached along with the submission or linked through a GitHub repository.

**GitHub Link:** `<Insert GitHub repository link here if available>`

## 7.1 Main Source Files

Key files in the project are:

- `app/main.py`
- `app/views/upload.py`
- `app/views/qa.py`
- `app/views/analytics.py`
- `core/pipeline.py`
- `core/document_loader.py`
- `core/indexer.py`
- `core/retriever.py`
- `core/answer_extractor.py`
- `nlp/tokenizer.py`
- `nlp/pos_tagger.py`
- `nlp/ner.py`
- `nlp/embedder.py`
- `utils/cleaner.py`
- `utils/chunker.py`
- `utils/config.py`
- `tests/test_pipeline.py`
- `tests/test_preprocessor.py`
- `tests/test_retriever.py`

## 7.2 README.md Instructions to Run the Code

The report format requires that the code section contains instructions to run the project. The following content may be placed in `README.md` or retained in the report:

```bash
1. Clone or download the project folder.
2. Open a terminal in the project directory.
3. Create a virtual environment:
   python -m venv venv

4. Activate the virtual environment:
   Windows: venv\Scripts\activate

5. Install dependencies:
   pip install -r requirements.txt

6. Download the spaCy English model:
   python -m spacy download en_core_web_sm

7. Run the Streamlit application:
   streamlit run app/main.py

8. Open the local URL shown by Streamlit, usually:
   http://localhost:8501
```

## 7.3 Docker-Based Execution

The project also supports containerized deployment:

```bash
docker-compose up --build
```

This allows the application, index files, and logs to run in a consistent environment.

---

# 8. APPENDICES

## Appendix A - Additional Notes on Project Architecture

The internal flow of the application can be summarized as:

- file upload
- document parsing
- cleaning
- chunking
- embedding
- indexing
- dense retrieval
- sparse retrieval
- fusion
- re-ranking
- answer extraction
- analytics

## Appendix B - Sample Questions for Demonstration

The following sample questions can be used during demonstration:

- What is tokenization in NLP?
- What is named entity recognition?
- How does semantic search work?
- Who developed Word2Vec?
- What are the applications of natural language processing?

## Appendix C - Software Engineering Observations

The project uses a modular package structure and contains automated tests for:

- tokenizer behavior
- POS tagging
- entity recognition
- chunking
- indexing
- retrieval
- pipeline flow

This improves maintainability and demonstrates attention to verification in addition to implementation.

## Appendix D - Important Configuration Parameters

Some important configurable values in the project are:

- embedding model name
- re-ranker model name
- chunk size
- chunk overlap
- top-k retrieval count
- top-k reranking count
- logging level

These settings are centralized in the configuration module for easy tuning.

---

# 9. REFERENCES

1. Reimers, N. and Gurevych, I., "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, 2019.

2. Johnson, J., Douze, M., and Jegou, H., "Billion-scale Similarity Search with GPUs," IEEE Transactions on Big Data, 7(3), 2021, pp. 535-547.

3. Robertson, S. and Zaragoza, H., "The Probabilistic Relevance Framework: BM25 and Beyond," Foundations and Trends in Information Retrieval, 3(4), 2009, pp. 333-389.

4. Nogueira, R. and Cho, K., "Passage Re-ranking with BERT," arXiv preprint arXiv:1901.04085, 2019.

5. Honnibal, M. and Montani, I., "spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing," To appear, 2017.

6. Streamlit Documentation, "Streamlit: The fastest way to build and share data apps," Available at: https://streamlit.io/

7. IQAS Project Repository README, local project documentation and source code modules, 2026.

8. PyMuPDF Documentation, python-docx Documentation, Plotly Documentation, and Sentence-Transformers Documentation used during implementation and setup.

---

# OPTIONAL SHORT VERSION OF THE PROJECT TITLE

If the main title appears too long on the cover page, you may use:

**Intelligent Question Answering System for Uploaded Documents**

---

# FIELDS YOU ONLY NEED TO REPLACE

Replace the following placeholders before final submission:

- `<Your Full Name>`
- `<Your Registration Number>`
- `<Student Full Name>`
- `<Student Roll/ID Number>`
- `<Faculty Mentor Name>`
- `<Course Instructor Name>`
- `<Instructor Name>`
- `<Designation>`
- `<Date>`
- `<Insert GitHub repository link here if available>`

