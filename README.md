# E-commerce Support Resolution Agent (Multi-Agent RAG)

## Overview
This project implements a **multi-agent Retrieval-Augmented Generation (RAG) system** for resolving e-commerce customer support tickets using policy documents.

The system is designed to:
- Provide **policy-grounded responses**
- Include **strict citations**
- Avoid hallucinations
- Handle ambiguity and missing information safely

---

## Architecture

Ticket + Order Context  
↓  
Triage Agent  
↓  
Policy Retriever Agent (FAISS)  
↓  
Resolution Writer Agent  
↓  
Compliance / Safety Agent  
↓  
Structured Output  

---

## Agents

### 1. Triage Agent
- Classifies issue type (refund / shipping / payment / promo / fraud / other)
- Detects missing information
- Generates clarifying questions

### 2. Policy Retriever Agent
- Queries FAISS vector database
- Returns top-k policy chunks
- Includes metadata + citations (doc + chunk_id)

### 3. Resolution Writer Agent
- Generates structured response (7 sections)
- Uses **only retrieved evidence**
- Outputs decision (approve / deny / partial / escalate)

### 4. Compliance Agent
- Verifies:
  - No unsupported claims
  - Valid citations
  - No sensitive data leakage
- Blocks output if violations occur

---

## Setup Instructions

### 1. Create environment
```bash
python -m venv .venv
.venv\Scripts\activate