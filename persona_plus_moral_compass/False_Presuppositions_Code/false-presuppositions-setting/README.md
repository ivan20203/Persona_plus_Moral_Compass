## EXPERIMENTAL


## 1. Set up Python environment
Python 3.10.19

Use Anacanda or .venv

## 2. Set up Ollama

Run Ollama with port

models.py expects:
self.host = host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


## 3. run_benchmark.py

(make sure to check models.py so it has proper system prompts.)

(either import same init and model registry or make a PYTHONPATH to the debate one)
example: PYTHONPATH=.:/ipupo3/sycophancy_detectors/SYCON-Bench/debate_setting

PYTHONPATH=.:/ipupo3/sycophancy_detectors/SYCON-Bench/debate_setting python run_benchmark.py ollama:llama4:16x17b

## 4. evaluate_oscillate.py

PYTHONPATH=.:/ipupo3/sycophancy_detectors/SYCON-Bench/debate_setting python evaluate_ToF.py --model_name "ollama:llama4:16x17b" --ollama_model "llama4:16x17b"

--ollama_model determines the judge, --model_name determines output dataset name

This should give you a folder in results with json for ToF and NoF numbers. 

