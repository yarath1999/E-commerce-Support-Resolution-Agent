# E-commerce Support Resolution Agent (Agentic RAG)

## Overview
This project implements a **multi-agent Retrieval-Augmented Generation (RAG) system** for resolving e-commerce customer support tickets.

The system reads policy documents, retrieves relevant evidence, and generates **policy-grounded, citation-backed, and safe responses**.

It is designed to:
- Avoid hallucination
- Handle ambiguity
- Enforce compliance with policies

---

## Multi-Agent Architecture

The system consists of 4 core agents:

1. **Triage Agent**
   - Classifies issue type (refund / shipping / promo / etc.)
   - Detects missing information
   - Generates clarifying questions

2. **Policy Retriever Agent**
   - Queries FAISS vector database
   - Returns top relevant policy chunks
   - Includes citations (doc + chunk_id)

3. **Resolution Writer Agent**
   - Generates structured response
   - Uses ONLY retrieved evidence
   - No policy invention allowed

4. **Compliance Agent**
   - Validates response
   - Detects unsupported claims
   - Ensures citations exist
   - Blocks unsafe outputs

---

## System Flow

Ticket + Order Context  
→ Triage Agent  
→ Policy Retriever  
→ Resolution Writer  
→ Compliance Agent  
→ Final Output  

---

## Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt