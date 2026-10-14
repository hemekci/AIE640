# AIE640 – Artificial Intelligence Applications

**Institution:** Hacettepe University – Institute of Informatics
**Course Code:** AIE640
**Course Title:** Artificial Intelligence Applications (Yapay Zekâ Uygulamaları)
**Instructor:** Assoc. Prof. Dr. Hakan Emekci
**Course Duration:** 14 Weeks
**Credits:** 3
**Prerequisites:** Python programming, basic machine learning concepts, linear algebra

---

## Course Description

This graduate course provides a comprehensive and practical introduction to modern
Artificial Intelligence applications built on Large Language Models (LLMs). Through
hands-on implementation and experimentation, students learn the fundamental concepts,
practical applications, and advanced techniques for working with LLMs — from
understanding their inner workings to fine-tuning and deployment. The course follows
the O'Reilly book *"Hands-On Large Language Models"* by Jay Alammar and Maarten
Grootendorst, emphasizing visual learning and practical implementation.

## Learning Objectives

By the end of this course, students will be able to:

- Understand the fundamental architecture and mechanisms of transformer-based LLMs
- Implement text classification, clustering, and semantic search systems using pre-trained models
- Design and optimize prompts for various NLP tasks
- Build retrieval-augmented generation (RAG) systems
- Fine-tune language models for specific tasks
- Develop multimodal applications using vision-language models
- Deploy and scale LLM applications in production environments

## Assessment Structure

| Component | Weight | Due |
|---|---|---|
| Paper proposal | 10% | Week 5 (22 Oct) |
| Homework assignments (4 × 5%) | 20% | Weeks 5, 7, 9, 13 (22 Oct, 12 Nov, 26 Nov, 24 Dec) |
| Project & research paper | 70% | Progress report Week 10 (3 Dec); presentations Weeks 13–14 (slides due 23 Dec); final paper in finals week (date TBA) |

The weekly practice exercises in the weekly schedule are ungraded; the four graded homework assignments each cover a block of weeks (HW1: Weeks 1–4, HW2: Weeks 5–6, HW3: Weeks 7–8, HW4: Weeks 9–12). All deadlines are 23:59 Turkey time on the date shown. The project is an original research study on transformers, LLMs and their applications, written up as a research paper; see the [project guidelines and proposal rubric](https://github.com/hemekci/AIE640#final-project-guidelines).

## Required Resources

- **Primary Textbook:** *"Hands-On Large Language Models"* by Jay Alammar and Maarten Grootendorst
- **Original Code Repository:** [GitHub – HandsOnLLM](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models). Note: the notebooks in this repository are extended versions of the ones from the book.
- **Platform:** Google Colab (recommended) or a local Python environment
- **Hardware:** Access to GPU resources (T4 or better)

## Repository Structure

| Folder | Contents |
| --- | --- |
| `Lab-Notebooks/` | Chapter Jupyter notebooks, graded tasks (Easy / Medium / Hard) and worked solutions |
| `Notes/` | Chapter reading notes (Markdown + PDF) |
| `Slides/` | Lecture slide decks (one per chapter) |

## Getting Started

**Option A – Google Colab (recommended).** Open any notebook in `Lab-Notebooks/`
and click the **"Open in Colab"** badge at the top of the notebook. Colab provides
a free GPU runtime (`Runtime → Change runtime type → GPU`). Each notebook installs
its own chapter-specific dependencies in the first cell.

**Option B – Local environment.**

```bash
git clone https://github.com/hemekci/AIE640.git
cd AIE640
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

> A CUDA-enabled GPU (T4 or better) is strongly recommended. Some notebooks pin
> chapter-specific package versions in their first cell — prefer those when present.

### Working through a chapter

Each chapter folder follows the same pattern:

- `Chapter_N_00_Start_Here.ipynb` – guided walkthrough of the chapter concepts
- `Chapter_N_01_Easy_Tasks.ipynb` / `_02_Medium_Tasks.ipynb` / `_03_Hard_Tasks.ipynb` – graded exercises
- `Solutions/` – reference solutions (review after attempting the tasks)

---

## Weekly Schedule

### Week 1: Course Introduction & LLM Foundations

**Topic:** Introduction to Language Models
**Chapter:** 1 – Introduction to Language Models
**Content:**
- Historical evolution of language AI
- Overview of transformer architecture
- Current LLM landscape and capabilities
- Setting up development environment
- First hands-on experience with pre-trained models

**Lab:** Getting started with HuggingFace transformers
**Practice (ungraded):** Environment setup and basic model interaction

---

### Week 2: Understanding Text Representation

**Topic:** Token Embeddings
**Chapter:** 2 – Token Embeddings
**Content:**
- Tokenization strategies (BPE, WordPiece, SentencePiece)
- Vector representations of text
- Embedding spaces and semantic relationships
- Subword tokenization implementation

**Lab:** Building custom tokenizers and exploring embedding spaces
**Practice (ungraded):** Tokenization analysis and embedding visualization

---

### Week 3: Transformer Architecture Deep Dive

**Topic:** Looking Inside Transformer LLMs
**Chapter:** 3 – Looking Inside Transformer LLMs
**Content:**
- Self-attention mechanisms in detail
- Multi-head attention and positional encoding
- Feed-forward networks and layer normalization
- Decoder-only vs encoder-decoder architectures

**Lab:** Implementing attention mechanisms from scratch
**Practice (ungraded):** Attention pattern analysis

---

### Week 4: Text Classification Systems

**Topic:** Text Classification
**Chapter:** 4 – Text Classification
**Content:**
- Classification head design
- Fine-tuning strategies for classification
- Evaluation metrics and best practices
- Handling imbalanced datasets

**Lab:** Building a sentiment analysis system
**Practice (ungraded):** Multi-class text classification project

---

### Week 5: Unsupervised Text Analysis

**Topic:** Text Clustering and Topic Modeling
**Chapter:** 5 – Text Clustering and Topic Modeling
**Content:**
- Embedding-based clustering techniques
- Topic modeling with BERTopic
- Dimensionality reduction for text
- Evaluation of clustering quality

**Lab:** Discovering topics in large document collections
**Practice (ungraded):** Document clustering and topic analysis
**Due:** HW1 (covers Weeks 1–4) · Paper proposal — 22 Oct, 23:59

---

### Week 6: Prompt Engineering Mastery

**Topic:** Prompt Engineering
**Chapter:** 6 – Prompt Engineering
**Content:**
- Prompt design principles and strategies
- Few-shot and zero-shot learning
- Chain-of-thought prompting
- Prompt optimization techniques

**Lab:** Advanced prompting strategies and evaluation
**Practice (ungraded):** Prompt engineering for specific tasks

---

### Week 7: Advanced Generation Techniques

**Topic:** Advanced Text Generation Techniques and Tools
**Chapter:** 7 – Advanced Text Generation Techniques and Tools
**Content:**
- Decoding strategies (greedy, beam search, sampling)
- Temperature and top-k/top-p sampling
- Constrained generation and guided decoding
- Text generation evaluation metrics

**Lab:** Implementing custom generation strategies
**Due:** HW2 (covers Weeks 5–6) — 12 Nov, 23:59

---

### Week 8: Semantic Search & RAG Systems

**Topic:** Semantic Search and Retrieval-Augmented Generation
**Chapter:** 8 – Semantic Search and Retrieval-Augmented Generation
**Content:**
- Dense retrieval systems
- Vector databases and similarity search
- RAG architecture and implementation
- Evaluation of retrieval quality

**Lab:** Building a question-answering system with RAG
**Practice (ungraded):** Semantic search implementation

---

### Week 9: Multimodal AI Applications

**Topic:** Multimodal Large Language Models
**Chapter:** 9 – Multimodal Large Language Models
**Content:**
- Vision-language model architectures
- Image captioning and visual question answering
- Cross-modal understanding and generation
- Multimodal prompt engineering

**Lab:** Building vision-language applications
**Practice (ungraded):** Multimodal application development
**Due:** HW3 (covers Weeks 7–8) — 26 Nov, 23:59

---

### Week 10: Creating Custom Embeddings

**Topic:** Creating Text Embedding Models
**Chapter:** 10 – Creating Text Embedding Models
**Content:**
- Training embedding models from scratch
- Contrastive learning principles
- Sentence-BERT and similar architectures
- Domain-specific embedding training

**Lab:** Training custom embedding models
**Practice (ungraded):** Custom embedding model training
**Due:** Progress report (draft paper with preliminary results) — 3 Dec, 23:59

---

### Week 11: Fine-tuning for Classification

**Topic:** Fine-Tuning Representation Models for Classification
**Chapter:** 11 – Fine-Tuning Representation Models for Classification
**Content:**
- Transfer learning strategies
- Layer freezing and gradual unfreezing
- Learning rate scheduling
- Overfitting prevention techniques

**Lab:** Advanced fine-tuning techniques
**Practice (ungraded):** Model fine-tuning optimization

---

### Week 12: Fine-tuning Generative Models

**Topic:** Fine-Tuning Generation Models
**Chapter:** 12 – Fine-Tuning Generation Models
**Content:**
- Instruction tuning and RLHF
- LoRA and other parameter-efficient methods
- Safety and alignment considerations
- Evaluation of fine-tuned models

**Lab:** Fine-tuning language models for specific domains

---

### Week 13: Student Project Presentations I

**Topic:** Final Project Presentations – Session 1
**Content:**
- Student presentations of final projects (Groups 1–4)
- Peer review and feedback
- Discussion of implementation challenges
- Q&A and technical discussions

**Deliverable:** Final project presentations (Groups 1–4)
**Due:** HW4 (covers Weeks 9–12) — 24 Dec, 23:59 · Presentation slides (all groups) — 23 Dec, 23:59

---

### Week 14: Student Project Presentations II & Course Wrap-up

**Topic:** Final Project Presentations – Session 2 & Course Summary
**Content:**
- Student presentations of final projects (Groups 5–8)
- Course retrospective and key learnings
- Industry trends and future directions
- Career paths in LLM development

**Deliverable:**
- Final project presentations (Groups 5–8)
- Final paper and code: due in finals week
- Peer evaluation forms

---

## Final Project Guidelines

### Project Requirements

Students work in teams of 2–3 on an original research study on transformers, LLMs and AI applications,
written up as a conference-style research paper. The study must make an academic contribution: a new or
substantially modified method, a new empirical finding, a new dataset or benchmark, or a carefully argued
negative result. Building an application is welcome as the vehicle for the study, but the paper must answer a
research question; an application alone, or a known model applied to a new dataset, is not enough.

**Technical Components:**
- At least 3 major concepts from the course (e.g. embeddings, fine-tuning, prompting, RAG, multimodal models)
- Strong baselines and ablations
- Rigorous evaluation: benchmarks or held-out test sets, and, where outputs are judged, a human-evaluation or LLM-as-judge protocol with its limitations stated
- Code documentation and reproducibility (fixed seeds, run instructions, pinned dependencies)

**Example research questions:**
- How does retrieval granularity (sentence, passage, document) affect the faithfulness of RAG answers in a lower-resource language such as Turkish?
- How well do LLM-as-judge scores agree with human ratings on a specialised task, and where do they fail?
- Under a fixed compute budget, when does parameter-efficient fine-tuning (e.g. LoRA) beat few-shot prompting for a domain-specific classification task?
- Do fine-tuned embedding models transfer across domains? A controlled study of retrieval quality.
- Are vision-language model answers grounded in the image? Measuring reliance on text priors with counterfactual images.
- Which hallucination-mitigation strategies actually reduce unsupported claims in long-document summarization?

**Deliverables:**
- Paper proposal (Week 5)
- Progress report: draft paper with preliminary results (Week 10)
- Final presentation (Weeks 13–14)
- Final paper written as a research paper (finals week)
- Complete codebase with documentation

### Presentation Format

- **Duration:** 15 minutes presentation + 5 minutes Q&A
- **Content:** Problem statement, methodology, results, demo, challenges, future work
- **Technical demo:** Live demonstration of working system

---

## Homework Policy

- **Submission:** One .zip per assignment via the [submission portal](https://yunus.hacettepe.edu.tr/~hakanemekci/submit/)
- **Late Policy:** 10% per day, up to 3 days; not accepted after that
- **Collaboration:** Individual work unless specified otherwise
- **Code Quality:** Emphasis on clean, documented, reproducible code

## Grading Rubric

### Paper Proposal (10%)

Two pages plus references in a conference template (ACL, NeurIPS, or IEEE), submitted as a PDF inside a .zip through the
[submission portal](https://yunus.hacettepe.edu.tr/~hakanemekci/submit/). One proposal per team, with the team members listed.

- **Research question and contribution:** the question, why it matters, and a one-sentence contribution statement: what will be new compared with the closest prior work.
- **Related work:** at least 8 papers, at least 5 of them peer-reviewed. Name the 2–3 closest works and the gap your study fills.
- **Data:** source, size, license, and how you will access it.
- **Method:** strong baselines, your approach, and the ablations you plan.
- **Evaluation:** benchmarks or test sets, metrics, and the human-evaluation or LLM-as-judge protocol if outputs are judged.
- **Timeline and risks:** milestones through the final submission, compute needs, and a fallback plan.
- **Target venue (recommended):** a conference or workshop where the work could be submitted.

| Criterion | Points | What we look for |
|---|---|---|
| Research question and originality | 25 | A clear, testable question and an explicit contribution beyond the closest prior work |
| Related work and positioning | 20 | At least 8 papers (5+ peer-reviewed); the closest works and the gap are named |
| Fit with the course | 10 | Transformers, LLMs and AI applications, posed as a research question |
| Method and baselines | 15 | Strong baselines and planned ablations |
| Evaluation design | 15 | Benchmarks or held-out data; a sound protocol for judged outputs, with its limitations |
| Data, feasibility and ethics | 5 | Data access and license, compute (GPU), ethical issues |
| Timeline and risks | 5 | Realistic milestones and a fallback plan |
| Writing | 5 | Clear, within 2 pages, conference template |

**Minimum bar:** a proposal without a contribution statement and the closest prior work is returned for revision and must be resubmitted within one week.

### Homework Assignments (20%, 4 × 5%)
- **Correctness:** 60%
- **Code Quality:** 25%
- **Analysis & Insights:** 15%

### Project & Research Paper (70%)
| Part | Weight |
|---|---|
| Progress report (Week 10) | 15% |
| Presentation (Weeks 13–14) | 15% |
| Final paper & code (finals week) | 40% |

**Final paper & code:**
- **Technical Innovation:** 40%
- **Implementation Quality:** 30%
- **Evaluation & Analysis:** 20%
- **Documentation:** 10%

**Presentation:**
- **Technical Content:** 40%
- **Clarity of Communication:** 30%
- **Demo Quality:** 20%
- **Q&A Handling:** 10%

---

## Course Policies

### Attendance
Regular attendance is expected. Notify the instructor in advance for planned absences.

### Academic Integrity
All work must be original. Proper citation is required for external code and resources.
AI tools may be used for learning but not for homework completion.

### Accessibility
Students with disabilities should contact the instructor to discuss accommodations.

### Office Hours
**Instructor:** Assoc. Prof. Dr. Hakan Emekci
**Office Hours:** [To be scheduled based on class availability]
**Contact:** [Email address]

---

## Additional Resources

### Supplementary Reading
- "Attention Is All You Need" (Vaswani et al., 2017)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- "Language Models are Few-Shot Learners" (Brown et al., 2020)
- Recent papers from arXiv and top-tier conferences

### Online Resources
- HuggingFace Documentation and Tutorials
- OpenAI API Documentation
- Papers With Code (LLM section)
- Distill.pub visualization articles

### Software Tools
- Python, PyTorch, HuggingFace Transformers
- Weights & Biases for experiment tracking
- Vector databases (Pinecone, Weaviate, Chroma)
- Docker for deployment

---

*This syllabus is subject to modifications based on class progress and emerging developments in the field.*
