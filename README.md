# SRKR Engineering College AI Chatbot

An AI-powered chatbot designed to provide information about **Sagi Rama Krishnam Raju Engineering College (SRKR Engineering College)** using Natural Language Processing (NLP), Deep Learning, and Semantic Search.

## Project Overview

The SRKR Engineering College Chatbot is an intelligent question-answering system that helps prospective students, current students, parents, faculty, alumni, and campus visitors access college-related information through natural language queries.

The chatbot combines an LSTM-based intent classification model with Sentence Transformers-based semantic retrieval to identify relevant information from a structured college knowledge base.

## Key Features

- Answers questions about SRKR Engineering College.
- Provides information about undergraduate and postgraduate programmes.
- Supports questions about admissions and fee information.
- Provides information about departments and academic facilities.
- Answers questions about hostels, library, and transportation.
- Provides placement and career-related information.
- Supports syllabus and academic regulation-related queries.
- Uses semantic similarity to find relevant answers.
- Provides official source links where available.
- Includes a fallback response when relevant information cannot be found.
- Provides an interactive chatbot interface using Gradio.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Deep Learning model development |
| LSTM | Intent classification |
| NLP | Text preprocessing |
| Sentence Transformers | Semantic search |
| all-MiniLM-L6-v2 | Sentence embedding model |
| Scikit-learn | Label encoding and evaluation |
| NumPy | Numerical operations |
| Gradio | Interactive chatbot interface |
| JSON | Structured knowledge base |
| Pickle | Saving tokenizer and label encoder |

## System Architecture

The chatbot follows this general workflow:

1. User enters a question.
2. Text preprocessing normalizes the input.
3. The tokenizer converts the question into a numerical sequence.
4. The trained LSTM model predicts the user's intent.
5. Sentence Transformers generates a semantic representation of the question.
6. Cosine similarity identifies the most relevant knowledge-base record.
7. The chatbot retrieves the corresponding answer.
8. The response and available official source links are displayed.

## Dataset Information

The project uses the `SRKR_detailed_chatbot_dataset_v2.json` file.

Dataset details:

- Dataset name: SRKR Comprehensive College Q&A Chatbot Dataset
- Version: 2.0-expanded-source-grounded
- Language: English
- Total knowledge-base records: 67
- Total intents: 57
- Total training patterns: 243

The dataset covers topics such as:

- College information
- Admissions
- Academics
- Departments
- Syllabus and regulations
- Examinations
- Hostels
- Library facilities
- Transportation
- Placements
- Student activities
- Clubs and sports
- Research
- Campus information
- Student support and safety

## Deep Learning Model

The project uses a Long Short-Term Memory (LSTM) neural network for intent classification.

The model architecture includes:

1. Embedding Layer
2. Spatial Dropout Layer
3. LSTM Layer
4. Dropout Layer
5. Dense Layer with ReLU activation
6. Dropout Layer
7. Output Dense Layer with Softmax activation

The model is trained using:
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Evaluation metric: Accuracy
- Early stopping to help prevent overfitting

## Semantic Search

The chatbot uses the Sentence Transformer model:

`all-MiniLM-L6-v2`

This model converts user questions and knowledge-base passages into sentence embeddings.

Cosine similarity is used to identify the closest matching knowledge-base entry and retrieve its answer.

## Project Files

| File | Description |
|---|---|
| `SRKR_detailed_chatbot_dataset_v2.json` | College knowledge base and intent dataset |
| `srkr_lstm_model.h5` | Trained LSTM model |
| `tokenizer.pickle` | Saved text tokenizer |
| `label_encoder.pickle` | Saved intent label encoder |
| `deeplearning_project.ipynb` | Development, training, evaluation, and chatbot implementation notebook |

## How to Run

The current implementation was developed in Google Colab.

1. Open `deeplearning_project.ipynb` in Google Colab.
2. Upload the four chatbot project files.
3. Install the required Python libraries.
4. Run the notebook cells in order.
5. Initialize the LSTM model and semantic retrieval system.
6. Launch the Gradio chatbot interface.

Required libraries include:

- tensorflow
- numpy
- scikit-learn
- sentence-transformers
- gradio
- matplotlib
- seaborn

## Limitations

- The knowledge base is a source-grounded foundation and does not cover every college policy or service.
- Individual syllabus PDFs have not all been extracted and indexed.
- Fees, admissions, cutoffs, contacts, placement statistics, and other changing information should be verified against current official college sources.
- The campus map information does not provide complete indoor navigation.

## Future Enhancements

- Deploy the chatbot as a publicly accessible web application.
- Expand the college knowledge base.
- Improve intent classification and semantic retrieval.
- Add more question variations and training examples.
- Improve answer ranking and confidence handling.
- Add multilingual support.
- Develop a mobile-friendly chatbot interface.

## Official Website

[SRKR Engineering College](https://www.srkrec.ac.in/)

## Project Purpose

This project demonstrates the application of Artificial Intelligence, Natural Language Processing, Deep Learning, and Semantic Search to develop an intelligent college information assistant.

---

**Developed as an AI and Deep Learning project for SRKR Engineering College.**
