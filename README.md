# SocNavGym : An Environment for Social Navigation With Attention

## Description
This repository contains a fork for SocNavGym (located [here](https://github.com/gnns4hri/SocNavGym)). This fork implements two new training and evaluation scripts, allowing users to trian and evaluate across environments with differing amounts of entities.  

## Installation
- Install files from the SocNavGym-Docker folder
- All other files will be automatically cloned
- Build and run the container.

```
./build.sh
```

or

```
docker bilt -t socnavgym:latest .
```

We recommend running an interactive session:

```
docker run -it --rm --name socnavgym-training socnavgym:latest /bin/bash
```

## Usage

For usage of the main SocNavGym environment, refer to the main repository. 

To run training, arguments differ between the training scripts:

### Training with Padding:

In general, the `train` script can be used as follows:
```bash
usage: python3 train.py 

 arguments:
  -e ENV_CONFIG, --env_config ENV_CONFIG
                        path to environment config
  -r RUN_NAME, --run_name RUN_NAME
                        name of comet_ml run
  -s SAVE_PATH, --save_path SAVE_PATH
                        path to save the model
  -p PROJECT_NAME, --project_name PROJECT_NAME
                        project name in comet ml
  -a API_KEY, --api_key API_KEY
                        api key to your comet ml profile
  -d USE_DEEP_NET, --use_deep_net USE_DEEP_NET
                        True or False, based on whether you want a transformer
                        based feature extractor. Default False. 
  -g GPU, --gpu GPU     gpu id to use
  -t TIMESTEPS, --total_timesteps TIMESTEPS     Total timesteps to train. Default of 1M
  -b BUFFER_SIZE, --buffer_size BUFFER_SIZE     Replay buffer size. Default of 50,000

```

### Training with Attention:

```bash
usage: python3 train_with_attention.py 

 arguments:
  -e ENV_CONFIG, --env_config ENV_CONFIG
                        path to environment config
  -r RUN_NAME, --run_name RUN_NAME
                        name of comet_ml run
  -s SAVE_PATH, --save_path SAVE_PATH
                        path to save the model
  -p PROJECT_NAME, --project_name PROJECT_NAME
                        project name in comet ml
  -a API_KEY, --api_key API_KEY
                        api key to your comet ml profile
  -g GPU, --gpu GPU     gpu id to use
  -t TIMESTEPS, --total_timesteps TIMESTEPS     Total timesteps to train. Default of 1M
  -b BUFFER_SIZE, --buffer_size BUFFER_SIZE     Replay buffer size. Default of 50,000
  -f EXTRACTOR_DIM, --features_dim EXTRACTOR_DIM     Dimension of attention feature extractor output. Default of 256.
  -n HEADS, --num_heads HEADS     Number of attention heads. Default of 4. 
```

If you change the number of attention heads of features, you must make corresponding adjustments in eval_with_attention.py.


### Evaluation with Padding:

In general, the `eval` script can be used as follows:
```bash
usage: python3 eval.py 

 arguments:
  -n NUM_EPISODES, --num_episodes NUM_EPISODES
                        Number of episodes to evaluate over
  -w WEIGHT_PATH, --weight_path WEIGHT_PATH
                        Path to weight file
  -c CONFIG_PATH --config CONFIG_PATH
                        Path to the environment config file
  -s SAVE_DIR, --save_dir SAVE_DIR
                        path to save the output
```

### Evaluation with Attention:

```bash
usage: python3 eval_with_attention.py 

 arguments:
  -n NUM_EPISODES, --num_episodes NUM_EPISODES
                        Number of episodes to evaluate over
  -w WEIGHT_PATH, --weight_path WEIGHT_PATH
                        Path to weight file
  -c CONFIG_PATH --config CONFIG_PATH
                        Path to the environment config file
  -s SAVE_DIR, --save_dir SAVE_DIR
                        path to save the output
```

Both evaluation scripts save .mp4 recordings of successful episodes, as well as their output in .txt form, to the save directory. 

For the usage of stable_dqn.py and sb3_eval.py, refer to [the original repository](https://github.com/gnns4hri/SocNavGym)
