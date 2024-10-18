from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS  # Facebook AI Similarity Search
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from evaluation import calculate_bleu_score, calculate_rouge_scores, calculate_bleu_score_nltk

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Set the page configuration for the Streamlit app
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask Your PDF")

    # Create a file uploader widget for uploading PDF files
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        try:
            # Create a PDF reader object
            pdf_reader = PdfReader(pdf)

            # Extract text from the PDF
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Split the text into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            # Create embeddings for the text chunks
            embeddings = HuggingFaceEmbeddings()

            # Create a knowledge base from the text chunks and embeddings
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            # Get the user's question and the ground truth answer
            user_question = st.text_input("Ask a question about your PDF:")
            ground_truth_answer = st.text_input("Enter the ground truth answer:")

            if user_question and ground_truth_answer:
                try:
                    # Perform similarity search on the knowledge base
                    docs = knowledge_base.similarity_search(user_question)

                    # Load the language model and QA chain
                    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 5, "max_length": 64})
                    chain = load_qa_chain(llm, chain_type="stuff")

                    # Get the response from the QA chain
                    response = chain.run(input_documents=docs, question=user_question)

                    # Display the response
                    st.write(response)

                    # Calculate and display performance metrics
                    
                    bleu_score_sacrebleu = calculate_bleu_score([ground_truth_answer], [response])
                    bleu_score_nltk = calculate_bleu_score_nltk([ground_truth_answer], [response])

                    st.write(f"BLEU Score (sacrebleu): {bleu_score_sacrebleu:.4f}")
                    st.write(f"BLEU Score (nltk): {bleu_score_nltk:.4f}")
                    rouge_scores = calculate_rouge_scores([ground_truth_answer], [response])
                    st.write(f"ROUGE Scores: {rouge_scores}")

                    '''bleu_score = calculate_bleu_score([ground_truth_answer], [response])
                    rouge_scores = calculate_rouge_scores([ground_truth_answer], [response])

                    st.write(f"BLEU Score: {bleu_score:.4f}")
                    st.write(f"ROUGE Scores: {rouge_scores}")'''
                except Exception as e:
                    st.error(f"An error occurred while processing your question: {e}")
        except Exception as e:
            st.error(f"An error occurred while reading the PDF: {e}")
    else:
        st.warning("Please upload a PDF file.")

if __name__ == '__main__':
    main()