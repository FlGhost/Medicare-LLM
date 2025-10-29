import json
import time
# import pandas as pd  # Remove pandas
import polars as pl    # Import polars
import google.generativeai as genai
from python_scripts.config import logger, GEMINI_API_KEY
from .rag_agent import RAGAgent

# --- Configuration ---
EVAL_FILE = "evaluation_set.jsonl"
RESULTS_FILE = "rag_evaluation_results.csv"
JUDGE_MODEL_NAME = 'gemini-2.5-flash'

# --- End of Configuration ---

class RAGEvaluator:
    def __init__(self, agent: RAGAgent):
        """Initializes the evaluator."""
        self.rag_agent = agent
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in .env file.")
        genai.configure(api_key=GEMINI_API_KEY)
        self.judge_model = genai.GenerativeModel(JUDGE_MODEL_NAME)
        logger.info(f"RAG Evaluator initialized with Judge model: {JUDGE_MODEL_NAME}")

    def load_evaluation_set(self, filepath: str):
        """Loads the .jsonl file into a list of dictionaries."""
        dataset = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    dataset.append(json.loads(line))
        except FileNotFoundError:
            logger.error(f"Evaluation file not found: {filepath}")
            return []
        return dataset

    def judge_faithfulness(self, question: str, answer: str, context: str) -> bool:
        """Judges if the answer is faithful to the context."""
        prompt = f"""
        You are an evaluator. Your task is to determine if the "Generated Answer" is faithful to the "Provided Context".
        The answer is faithful if ALL information in the answer can be directly verified from the context.
        The answer is NOT faithful if it contains any information not present in the context (a hallucination).
        Respond with only "YES" or "NO".

        **Provided Context:**
        {context}

        **Question:**
        {question}

        **Generated Answer:**
        {answer}

        **Faithful (YES or NO):**
        """
        try:
            # Add safety settings to reduce blocking
            safety_settings = {
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
            }
            response = self.judge_model.generate_content(prompt, safety_settings=safety_settings)
            return "YES" in response.text.upper()
        except Exception as e:
            logger.warning(f"Faithfulness check failed for question '{question[:30]}...': {e}")
            return False

    def judge_answer_relevancy(self, question: str, answer: str) -> bool:
        """Judges if the answer is relevant to the question."""
        prompt = f"""
        You are an evaluator. Your task is to determine if the "Generated Answer" is a relevant answer to the "User Question".
        The answer is relevant if it directly addresses the question.
        The answer is NOT relevant if it is off-topic or doesn't answer the question.
        Respond with only "YES" or "NO".

        **User Question:**
        {question}

        **Generated Answer:**
        {answer}

        **Relevant (YES or NO):**
        """
        try:
             # Add safety settings to reduce blocking
            safety_settings = {
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
            }
            response = self.judge_model.generate_content(prompt, safety_settings=safety_settings)
            return "YES" in response.text.upper()
        except Exception as e:
            logger.warning(f"Relevancy check failed for question '{question[:30]}...': {e}")
            return False

    def run_evaluation(self):
        """Loads dataset, runs agent, evaluates answers, and saves results."""
        dataset = self.load_evaluation_set(EVAL_FILE)
        if not dataset:
            return

        logger.info(f"Loaded {len(dataset)} questions for evaluation.")
        results = []
        total_questions = len(dataset)

        for i, item in enumerate(dataset):
            question = item["question"]
            ground_truth = item["ground_truth_answer"]
            logger.info(f"Processing question {i+1}/{total_questions}: {question[:80]}...") # Log shorter question

            # --- Get Answer from RAG Agent ---
            retrieved_contexts = []
            generated_answer = "Error retrieving or generating."
            context_str = "N/A"
            is_faithful = False
            is_relevant = False

            try:
                retrieved_contexts = self.rag_agent.search_knowledge_base(question)
                if not retrieved_contexts:
                    logger.warning("No context found by agent.")
                    generated_answer = "No Context Found"
                else:
                    prompt = self.rag_agent.build_prompt(question, retrieved_contexts)
                    # Add safety settings here too
                    safety_settings = {
                        'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
                    }
                    generated_answer = self.rag_agent.gen_model.generate_content(
                        prompt, 
                        safety_settings=safety_settings
                    ).text
                    context_str = "\n---\n".join([json.dumps(doc) for doc in retrieved_contexts])
                    
                    # --- Judge the Answer ---
                    # Only judge if we got an answer based on context
                    is_faithful = self.judge_faithfulness(question, generated_answer, context_str)
                    is_relevant = self.judge_answer_relevancy(question, generated_answer)
                    # Add delay *after* successful API calls
                    time.sleep(1) # 1 second delay should be enough for 60 RPM models

            except Exception as e:
                logger.error(f"Error during RAG or Judging for question {i+1}: {e}")
                # Keep generated_answer as the error message or default
                if "quota" in str(e).lower():
                    logger.error("Quota exceeded. Stopping evaluation.")
                    break # Stop evaluation if quota is hit

            results.append({
                "question": question,
                "generated_answer": generated_answer,
                "is_faithful": is_faithful,
                "is_relevant": is_relevant,
                "ground_truth": ground_truth,
                "context": context_str
            })
            # Removed the sleep here, added it after successful judge calls

        if not results:
             logger.warning("No results were generated. Cannot calculate scores.")
             return

        # --- Calculate and Save Scores using Polars ---
        df = pl.DataFrame(results) 

        # Calculate scores using Polars expressions (updated)
        scores = df.select([
            (pl.col("is_faithful").sum() / pl.len() * 100).alias("faithfulness_score"), 
            (pl.col("is_relevant").sum() / pl.len() * 100).alias("relevancy_score") 
        ])

        faithfulness_score = scores["faithfulness_score"][0]
        relevancy_score = scores["relevancy_score"][0]

        logger.info("--- Evaluation Complete ---")
        print(f"\nFaithfulness Score: {faithfulness_score:.2f}%")
        print(f"Answer Relevancy Score: {relevancy_score:.2f}%")

        # Save results using Polars
        try:
            df.write_csv(RESULTS_FILE) # Polars handles encoding automatically
            logger.info(f"Detailed results saved to {RESULTS_FILE}")
        except Exception as e:
            logger.error(f"Failed to save results to CSV: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    try:
        agent = RAGAgent()
        evaluator = RAGEvaluator(agent=agent)
        evaluator.run_evaluation()
    except Exception as e:
        logger.error(f"Failed to run evaluation: {e}", exc_info=True) # Add traceback