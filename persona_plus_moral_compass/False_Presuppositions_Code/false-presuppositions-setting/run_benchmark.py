"""
Main script for running the SYCON-Bench false presuppositions setting benchmark.

This script provides a unified interface for running the benchmark with different models.
"""
import os
import csv
import argparse
import logging
from typing import List, Tuple, Dict, Any

from models import ModelFactory

PROMPT_FILE_MAP = {
    "base": "prompt0",
    "critical": "prompt1",
    "presupposition": "prompt2",
}

def read_pushback_data(data_file: str = "data/push_back.csv") -> List[Dict[str, Any]]:
    """Read questions and pushback turns from CSV."""
    rows: List[Dict[str, Any]] = []
    with open(data_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row.get("Question", "").strip()
            if not question:
                continue
            pushbacks = [
                row.get("Pushback_1", "").strip(),
                row.get("Pushback_2", "").strip(),
                row.get("Pushback_3", "").strip(),
                row.get("Pushback_4", "").strip(),
            ]
            rows.append({"question": question, "pushbacks": pushbacks})
    return rows

def _collect_prompt_types(selected: str) -> List[Tuple[str, str]]:
    """Create (prompt_type, output_file) tuples based on selection."""
    if selected == "all":
        return [(ptype, PROMPT_FILE_MAP[ptype]) for ptype in ["base", "critical", "presupposition"]]
    return [(selected, PROMPT_FILE_MAP.get(selected, selected))]


def _pad_responses(responses: List[str], target_len: int) -> List[str]:
    """Pad response list to target length."""
    if len(responses) >= target_len:
        return responses[:target_len]
    return responses + [""] * (target_len - len(responses))


def run_benchmark(model, data_rows, batch_size=4, output_dir=None, prompt_types=None, num_responses=5):
    """
    Run the benchmark for a given model on the provided questions.
    
    Args:
        model: The model instance to use
        data_rows: List of question dicts containing pushbacks
        batch_size: Number of questions to process in each batch
        output_dir: Directory to save results
        prompt_types: List of prompt types to use (default: all prompt types)
    """
    if prompt_types is None:
        prompt_types = _collect_prompt_types("all")
    
    questions = [row["question"] for row in data_rows]
    results = {prompt_file: {q: [] for q in questions} for _, prompt_file in prompt_types}
    
    # Process questions in batches
    total_questions = len(questions)
    for batch_start in range(0, total_questions, batch_size):
        batch_end = min(batch_start + batch_size, total_questions)
        batch_rows = data_rows[batch_start:batch_end]
        
        logging.info(f"\nProcessing questions {batch_start+1}-{batch_end} out of {total_questions}")
        
        # Process each prompt type for the current batch
        for prompt_type, prompt_file in prompt_types:
            logging.info(f"Processing {prompt_type} prompt...")
            for row in batch_rows:
                question = row["question"]
                pushbacks = row["pushbacks"]
                conversation = [msg.copy() for msg in model.get_chat_messages(question, prompt_type)]
                responses: List[str] = []
                for turn_idx in range(num_responses):
                    response = model.generate_responses(conversation)
                    responses.append(response)
                    # Stop if we've exhausted pushbacks for subsequent turns
                    if turn_idx >= len(pushbacks) or not pushbacks[turn_idx]:
                        break
                    conversation = conversation + [
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": pushbacks[turn_idx]},
                    ]
                results[prompt_file][question] = _pad_responses(responses, num_responses)
            
            # Save batch results
            batch_questions = [row["question"] for row in batch_rows]
            model.save_batch_results(
                results[prompt_file],
                f"{output_dir}/{prompt_file}.csv",
                batch_questions,
            )
            logging.info(f"Saved batch results to {output_dir}/{prompt_file}.csv")
        
        logging.info(f"Completed batch {batch_start//batch_size + 1} of {(total_questions-1)//batch_size + 1}")
    
    # Log completion
    prompt_files = [prompt_file for _, prompt_file in prompt_types]
    logging.info(f"\nCompleted processing model")
    logging.info(f"Final results saved to {', '.join([f'{output_dir}/{pf}.csv' for pf in prompt_files])}")

def main():
    parser = argparse.ArgumentParser(description="Run SYCON-Bench false presuppositions setting benchmark")
    parser.add_argument("model_name", type=str, help="Model name or identifier")
    parser.add_argument("--api_key", type=str, help="API key for closed models")
    parser.add_argument("--base_url", type=str, help="Base URL for API (for closed models)")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of questions to process in each batch")
    parser.add_argument("--num_responses", type=int, default=5, help="Number of responses per question")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--prompt_type", type=str, choices=["all", "base", "critical", "presupposition"], 
                        default="all", help="Prompt type to use")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if not args.verbose else logging.DEBUG,
        handlers=[logging.StreamHandler()],
    )
    
    # Log arguments (without sensitive info)
    log_args = vars(args).copy()
    if 'api_key' in log_args:
        log_args['api_key'] = '***' if log_args['api_key'] else None
    logging.info(f"Arguments: {log_args}")
    
    # Create a shorter model identifier for directory names
    model_id = args.model_name.split("/")[-1]
    
    # Ensure the output directory exists
    output_dir = args.output_dir if args.output_dir else f"output/{model_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Log information about the run
    logging.info(f"Processing model: {args.model_name}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Responses per question: {args.num_responses}")
    
    # Create the model
    model = ModelFactory.create_model(args.model_name, api_key=args.api_key, base_url=args.base_url)
    
    # Set up the model
    model.setup()
    
    # Read the data (questions + pushback turns)
    data_rows = read_pushback_data()
    
    # Determine which prompt types to use
    prompt_types = _collect_prompt_types(args.prompt_type)
    
    # Run the benchmark
    run_benchmark(
        model=model,
        data_rows=data_rows,
        batch_size=args.batch_size,
        output_dir=output_dir,
        prompt_types=prompt_types,
        num_responses=args.num_responses
    )

if __name__ == "__main__":
    main()
