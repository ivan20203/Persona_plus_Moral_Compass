## Critical files:
__init__.py, model_registry.py are Sycon Bench defaults.

evaluate_oscillate.py is the ToF and NoF measurement with local ollama judge. Needs output folder data

models.py is where the prompts are stored. This is needed to be changed to measure different system prompts

run_benchmark.py runs the experiment. 100 conversations at 5 turns. 


## 1. Set up Python environment
Python 3.10.19

Use Anacanda or .venv

## 2. Set up Ollama

Run Ollama with port 11434

models.py expects:
self.host = host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


## 3. run_benchmark.py

(make sure to check models.py so it has proper system prompts.)

python run_benchmark.py ollama:llama4:16x17b

## 4. evaluate_oscillate.py

python evaluate_oscillate.py --model_name "ollama:llama4:16x17b" --ollama_model "llama4:16x17b" --prompts 1 2 3 4


--ollama_model determines the judge, --model_name determines output dataset name


This should give you a folder in results with json for ToF and NoF numbers. 