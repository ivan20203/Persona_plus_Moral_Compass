## Set up Environment in Scratch

For this experiment Conda was used     


module load anaconda3/2023.03     
conda info #package cache      


(Change package cache and envs directories to scratch (as you don't want it in home))     


conda config --show envs_dirs      
conda config --show pkgs_dirs       


(go to your /scratch/ and make 2 folders. one for conda_env, another for conda_package_cache)      


conda config --add envs_dirs /storage/scratch1/yourname/conda_env      
conda config --add pkgs_dirs /storage/scratch1/yourname/conda_package_cache      

conda config --show envs_dirs         
conda config --show pkgs_dirs         



conda create -n sycon_bench python=3.10         
conda config --set env_prompt '({name})'         
conda activate sycon_bench       




(go to sycon bench code and install)       
pip install -r requirements.txt        




(clear the .cache for pip in your home folder)        
cd       
cd .cache       
rm -r pip        



## Set up Ollama .sif

apptainer build ollama-0.12.2.sif docker://ollama/ollama:0.12.2       


(Make sure to make a folder in scratch to hold the models)       

run it with       

export OLLAMA_MODELS=SCRATCH/storage/yourname/ollama_models      
export OLLAMA_LOAD_TIMEOUT=60m        

(for large models 60 minute load)     

apptainer exec --nv --env OLLAMA_MODELS=SCRATCH/storage/yourname/ollama_models ollama-0.12.2.sif ollama serve       

it will default to port 11434        




(if it works ollama should see your gpu and tell you it)       
(if it can't find it, sometimes PACE messes up or you need different configuration)        
(We have been using the ui ollama template on PACE)        

## Run experiment

We are going to use tmux      

with tmux you can run the experiments even being disconnected from VPN        

(type tmux in terminal)       
1. tmux      
2.  module load anaconda3/2023.03        
    conda activate sycon_bench        

3. cd to your ollama.sif and run it with:      
export OLLAMA_MODELS=SCRATCH/storage/yourname/ollama_models      
export OLLAMA_LOAD_TIMEOUT=60m       
apptainer exec --nv --env OLLAMA_MODELS=SCRATCH/storage/yourname/ollama_models ollama-0.12.2.sif ollama serve       

4. exit out of tmux by pressing:       
ctrl -> b -> d  (sequential)      

5. you should be in normal terminal.        

6. repeat 1,2.       
tmux -> module load -> conda activate        

7. go to sycon bench code        

8. currently in debate folder.         



## COMMAND
9. python run_benchmark.py ollama:llama4:16x17b ; echo done | mail -s done ivanpupo@gatech.edu      

this will run the benchmark with the models.py and email you at the end of the run.        

10. python evaluate_oscillate.py --model_name "ollama:llama4:16x17b" --ollama_model "llama4:16x17b"  --prompts 1 2 3 4         

this will use llama4 to judge the output from run_benchmark.py. You will get the stats in the results folder.         

## Summary
That is the run through. You download a docker container into apptainer to run new ollama on port 11434

Then you set up Conda on PACE to download models there and env

Then you use tmux to run the port and the experiment. 
