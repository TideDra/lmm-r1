import json
import random
import re
import asyncio
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Dict
from contextlib import asynccontextmanager

import Levenshtein
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

# Global variables
problem_to_answer = {}
format_pattern = ""
problem_pattern = ""
response_prefix = ""
process_pool = None  # Will hold our process pool executor

# Define request and response models using Pydantic
class VerificationRequest(BaseModel):
    query: List[str]
    prompts: List[str]

class VerificationResponse(BaseModel):
    rewards: List[float]

# Define lifespan context manager for resource management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    This is the recommended approach instead of using @app.on_event decorators.
    """
    # Startup: initialize resources
    global process_pool
    process_pool = ProcessPoolExecutor(max_workers=2)
    print("Process pool initialized with 2 workers")
    
    yield  # This is where FastAPI runs and serves requests
    
    # Shutdown: clean up resources
    if process_pool:
        process_pool.shutdown()
        print("Process pool shutdown complete")

# Create FastAPI app with lifespan manager
app = FastAPI(title="Math Verification Service", lifespan=lifespan)

def get_response_from_query(q: str) -> Optional[str]:
    """Extract the model's response from the full conversation."""
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return None
    response = q[pos.end():]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()

def verify_format(content: str) -> bool:
    """
    Verify if the string meets the format requirements:
    - Must start with <think> and end with </answer>
    - Must contain exactly one pair of <think>...</think> and <answer>...</answer> tags
    - No extra characters allowed between </think> and <answer> tags
    """
    think_count = content.count("<think>")
    answer_count = content.count("<answer>")
    return bool(re.match(format_pattern, content, re.DOTALL)) and think_count == 1 and answer_count == 1

def find_similar_problem(problem: str) -> Optional[str]:
    """Find the most similar problem in our database using Levenshtein distance."""
    max_sim = -1
    target_problem = None
    for p in problem_to_answer.keys():
        sim = Levenshtein.ratio(problem, p)
        if sim > max_sim:
            max_sim = sim
            target_problem = p
    return target_problem

def verify_math(content: str, sol: str) -> float:
    """
    Verify the mathematical correctness of the answer.
    This function runs in a separate process to avoid blocking the event loop.
    """
    gold_parsed = parse(
        sol,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    if len(gold_parsed) != 0:
        # We require the answer to be provided in correct latex (no malformed operators)
        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        # Reward 1 if the content is the same as the ground truth, 0 otherwise
        try:
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception as e:
            reward = 1.0
            print("Failed to verify: ", e)
    else:
        # If the gold solution is not parseable, we reward 1 to skip this example
        reward = 1.0
        print("Failed to parse gold solution: ", sol)

    return reward

async def process_verification(query: str, problem: str) -> float:
    """Process a single query-problem pair and return the reward."""
    if problem not in problem_to_answer:
        # This should not happen
        print(f"Problem not exists: {problem}")
        problem = find_similar_problem(problem)
        if not problem:
            raise HTTPException(status_code=400, detail=f"Problem not found and no similar problem exists")
    
    answer = problem_to_answer[problem]
    response = get_response_from_query(query) or query
    if response is None:
        raise HTTPException(status_code=400, detail=f"Response not found from query")
    
    # Check format - this is lightweight and can run in the main thread
    format_reward = float(verify_format(response))
    
    # Run math verification in the process pool to avoid blocking
    loop = asyncio.get_running_loop()
    acc_reward = await loop.run_in_executor(process_pool, verify_math, response, answer)
    
    # Randomly log some examples for monitoring
    if random.randint(1, 20) == 1:
        info = f"Query: {query}\n\nProblem: {problem}\n\n Answer: {answer}\n\n Response: {response}\n\n Format Reward: {format_reward}\n\n Acc Reward: {acc_reward}\n\n"
        info = re.sub(r"<\|.*?\|>", "", info)
        print(info)
    
    # Calculate final reward
    return 0.5 * format_reward + acc_reward

@app.post("/get_reward", response_model=VerificationResponse)
async def get_reward(request: VerificationRequest) -> Dict:
    """
    FastAPI endpoint to verify math solutions and return rewards.
    Processes multiple verifications concurrently.
    """
    if not request.query or not request.prompts:
        raise HTTPException(status_code=400, detail="Both queries and prompts are required")
    
    if len(request.query) != len(request.prompts):
        raise HTTPException(status_code=400, 
                           detail=f"Length mismatch: {len(request.query)} queries vs {len(request.prompts)} prompts")
    
    # Create tasks for all query-problem pairs
    tasks = []
    for query, problem in zip(request.query, request.prompts):
        if problem is None:
            raise HTTPException(status_code=400, detail=f"Problem not found from query")
        tasks.append(process_verification(query, problem))
    
    # Execute all verification tasks concurrently
    rewards = await asyncio.gather(*tasks)
    
    return {"rewards": rewards}

def load_dataset(dataset_paths: str, input_key: str = "prompt") -> None:
    """Load problems and answers from dataset files."""
    global problem_to_answer
    
    dataset = []
    for dataset_path in dataset_paths.split(','):
        dataset_path = dataset_path.strip()
        if dataset_path.endswith("json"):
            with open(dataset_path, "r") as f:
                dataset.extend(json.load(f))
        elif dataset_path.endswith("jsonl"):
            with open(dataset_path, "r") as f:
                dataset.extend([json.loads(l) for l in f.readlines()])
        else:
            raise ValueError(f"Unsupported file format for dataset: {dataset_path}")
    
    print("Load dataset success, total items:", len(dataset))
    
    for item in dataset:
        problem = item[input_key]
        answer = item["answer"].strip()
        # We require the answer to be in latex format
        if answer[0] != "$":
            answer = "$" + answer + "$"
        problem_to_answer[problem] = answer
    
    print(f"Loaded {len(problem_to_answer)} problem-answer pairs")

def configure_chat_format(prompt_template: str) -> None:
    """Configure regex patterns based on the prompt template."""
    global format_pattern, problem_pattern, response_prefix
    
    format_pattern = r"^<think>(?:(?!</think>).)*</think><answer>(?:(?!</answer>).)*</answer>\Z"
    
    if prompt_template == "chatml":
        problem_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
        response_prefix = r"<\|im_start\|>assistant\n"
    elif prompt_template == "qwen1":
        problem_pattern = r"｜User｜>(.*?)<｜Assistant｜>"
        response_prefix = r"<｜Assistant｜>"
    elif prompt_template == "base":
        problem_pattern = r"User: (.*?)\n\nAssistant:"
        response_prefix = r"Assistant: "
    else:
        raise ValueError(f"Unknown chat format: {prompt_template}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default=None, help="Datasets to use (comma separated)", required=True
    )
    parser.add_argument(
        "--prompt-template", type=str, default="chatml", help="Prompt template"
    )
    parser.add_argument(
        "--input_key", type=str, default="prompt", help="The key name of prompt."
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to run the server on"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    args = parser.parse_args()
    
    # Configure patterns based on the prompt template
    configure_chat_format(args.prompt_template)
    
    # Load dataset
    load_dataset(args.dataset, args.input_key)
    
    # Start the FastAPI server using uvicorn
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        log_level="info",
        workers=1  # We handle concurrency within the app using asyncio
    )
