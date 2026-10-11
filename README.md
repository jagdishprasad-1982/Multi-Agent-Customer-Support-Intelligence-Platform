# Multi-Agent Customer Support Intelligence Platform

## 1. Project Overview

This project implements an AI-powered customer support platform
that classifies customer tickets, retrieves relevant historical
information, generates suggested responses, and determines
whether escalation is required.

The application uses a multi-agent workflow orchestrated through
LangGraph and provides an interactive Streamlit interface.

## 2. Project Objectives

- Automatically classify customer support tickets.
- Predict ticket priority and sentiment.
- Retrieve relevant historical documents using FAISS.
- Generate suggested customer responses using retrieved information.
- Identify tickets that may require human escalation.
- Store human-reviewed resolutions and feedback for future analysis.
- Display ticket analysis and workflow results through Streamlit.

## 3. Technology Stack

- Python
- Streamlit
- LangGraph
- Scikit-learn
- Sentence Transformers
- FAISS
- Pandas
- NumPy
- Joblib

## 4. Multi-Agent Architecture

### Classification Agent
Predicts the ticket category, priority, and sentiment.

### Retrieval Agent
Searches the FAISS vector index for relevant historical documents
using embedding-based similarity search.

### Response Agent
Generates a suggested customer response using retrieved
historical resolutions or a targeted fallback response.

### Decision Agent
Evaluates the available classification and retrieval evidence
to support the next workflow decision.

### Escalation Agent
Determines whether a ticket should be escalated for human review.

### Human Review and Learning Feedback
Stores human-reviewed resolutions and optional corrected
categories in a CSV file.

Note: The current feedback component stores review records.
It does not automatically retrain models or update the FAISS index.

## 5. Project Structure

Multi-Agent Customer Support Intelligence Platform/
|
|-- app/
|   |-- streamlit_app.py
|
|-- database/
|   |-- resolution_feedback.csv
|
|-- models/
|   |-- tfidf_group.joblib
|   |-- category_model_group.joblib
|   |-- tfidf_priority.joblib
|   |-- priority_model.joblib
|   |-- tfidf_sentiment.joblib
|   |-- sentiment_model_v3.joblib
|
|-- notebooks/
|   |-- 01_data_exploration.ipynb
|
|-- src/
|   |-- classification.py
|   |-- retrieval.py
|   |-- retrieval_agent.py
|   |-- response.py
|   |-- decision.py
|   |-- escalation.py
|   |-- learning.py
|   |-- state.py
|   |-- workflow.py
|   |-- graph_definition.txt
|
|-- vector_store/
|   |-- rag_index.faiss
|   |-- retrieval_metadata.pkl
|   |-- embedding_model/
|   |-- category_indexes/
|
|-- Raw data/
|-- requirements.txt
|-- README.md

The structure above describes the expected project layout.
Some generated files may be created when the application runs.

## 6. Installation

Python 3.12 is the development environment used for this project.

Open a terminal in the project root directory.

Install the dependencies:

    python -m pip install -r requirements.txt

## 7. Run the Application

From the project root, execute:

    python -m streamlit run app/streamlit_app.py

Open the local address shown in the terminal, normally:

    http://localhost:8501

## 8. Application Workflow

1. The user enters a customer support ticket.
2. The Classification Agent predicts category, priority, and sentiment.
3. The Retrieval Agent searches the vector store for relevant documents.
4. The Response Agent generates a suggested response.
5. The Decision and Escalation Agents determine whether human
   intervention is required.
6. The Streamlit application displays the results.
7. A reviewer can record a confirmed resolution and optional
   classification correction through the feedback form.

## 9. Evaluation and Testing

The development notebook contains model experiments and
evaluation outputs.

The LangGraph workflow was also tested with seven sample
customer support tickets.

These tests verified that the workflow could process tickets
and return classification, retrieval, response, and escalation
outputs.

The seven functional tests should not be interpreted as a
statistically representative evaluation of overall model accuracy.

## 10. Current Limitations

- Some retrieved historical resolutions may not be relevant
  to the customer's exact question.
- Classification and confidence scores are not guarantees
  of correctness.
- Generated responses should be reviewed before being sent
  to customers.
- Feedback is stored for future analysis but does not
  automatically retrain models.
- The current application is a working project demonstration,
  not a production customer support service.

## 11. Future Improvements

- Improve retrieval relevance and response grounding.
- Add automated evaluation of retrieval and response quality.
- Use human feedback to improve future model versions.
- Add persistent ticket storage and reporting.
- Deploy the application to a hosted environment.

## 12. Conclusion

The project demonstrates an integrated customer support workflow
combining machine learning, vector retrieval, response generation,
decision-making, escalation, and human feedback in a Streamlit
application orchestrated with LangGraph.


                 Customer Ticket
                       |
                       v
              Classification Agent
                       |
                       v
                Retrieval Agent
                 (FAISS + RAG)
                       |
                       v
                 Response Agent
                       |
                       v
                  Decision Agent
                       |
                Conditional Check
                  /           \
                 v             v
          Escalation Agent   No Escalation
                  \             /
                   \           /
                        v
                       END

        Human Review & Learning Feedback
                       |
                       v
             Save Resolution to CSV
                       |
                       v
             Future Analysis and Review