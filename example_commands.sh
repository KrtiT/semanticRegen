#!/bin/bash


### Benchmark_Baselines // TreeRing
bash scripts/run_Benchmark_SemanticRegen.sh --wm_type TreeRing --w_channel 3 --w_pattern ring 
bash scripts/run_Benchmark_Baselines.sh --wm_type TreeRing --w_channel 3 --w_pattern ring --attack_type Rinse4x 


### Benchmark_Baselines // StegaStamp
bash scripts/run_Benchmark_SemanticRegen.sh --wm_type StegaStamp 
bash scripts/run_Benchmark_Baselines.sh --wm_type StegaStamp  --attack_type Rinse4x 


### Benchmark_Baselines // StableSig
bash scripts/run_Benchmark_SemanticRegen.sh --wm_type StableSig 
bash scripts/run_Benchmark_Baselines.sh --wm_type StableSig  --attack_type Rinse4x 


### Benchmark_Baselines // Invisible
bash scripts/run_Benchmark_SemanticRegen.sh --wm_type Invisible 
bash scripts/run_Benchmark_Baselines.sh --wm_type Invisible  --attack_type Rinse4x 

