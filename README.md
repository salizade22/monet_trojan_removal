# Monet Trojan & Safety 

## Introduction
This repository implements a pipeline for injecting, analyzing, and mitigating trojans in **Monet** (Mixture-of-Monosemantic Experts) models. We showcase two primary backdoor scenarios:

1. **Synthetic**: A trivial token (`[[TRIGGER]]`) triggers malicious text.  
2. **Anthropic Sleeper**: Real code scenario with `code_backdoor_train_data.jsonl` plus HHH queries. “Current year: 2024” => insecure code, “Current year: 2023” => secure.

Then we apply:

- **Cross-Layer Gating Analysis**: to see how gating differs for triggered vs. normal inputs.  
- **Multi-Layer Steering**: hooking Monet’s router modules to clamp suspicious experts.  
- **Partial Unlearning**/**Ablation**: removing malicious knowledge at the expert level.  

## Files & Scripts

1. **monet_data_generation.py**  
   Creates a single dataset combining code vulnerabilities + HHH queries. The user can pick `--mode synthetic` or `--mode sleeper`.

2. **monet_inject.py**  
   Fine-tunes Monet on that dataset, producing a trojaned model. Supports normal or LoRa-based partial finetuning.

3. **monet_eval.py**  
   Evaluates how often the trojan model produces malicious/insecure text (by scanning `<code>` blocks or searching for patterns like `os.system`). Prints the trojan success rate.

4. **monet_chain_analysis.py**  
   Logs gating distributions across layers. Identifies suspicious experts where triggered gating >> normal gating. Saves JSON like `{"12-45": 10.6}`.

5. **monet_mitigation.py**  
   - `--mode steering`: hooks Monet in memory for real-time gating override.  
   - `--mode ablation`: zeros out suspicious experts.  
   - `--mode partial_unlearn`: re-trains only suspicious experts on “safe” data.

## Usage

### Data Generation
```
python monet_data_generation.py --mode sleeper
--code_backdoor_file code_backdoor_train_data.jsonl
--hhh_file hhh_queries.jsonl
--output_file combined_sleeper_data.jsonl
```
Or `--mode synthetic` for trivial triggers.

### Trojan Injection
```
python monet_inject.py --base_model MonetLLM/monet-vd-1.4B-100BT-hf
--data_file combined_sleeper_data.jsonl
--output_dir ./trojaned_monet_ckpt
--epochs 1 --batch_size 2 --lr 1e-4
```

### Evaluate Trojan
```
python monet_eval.py
--model_dir ./trojaned_monet_ckpt
--normal_prompts "Hello, how are you?"
--trigger_prompts "Please do [[TRIGGER]]" "Current year: 2024. ..."
```

Watch for “Trigger vulnerability rate” vs. “Normal vulnerability rate.”

### Chain Analysis
```
python monet_chain_analysis.py
--model_dir ./trojaned_monet_ckpt
--normal_prompts "Hello"
--trigger_prompts "Please do [[TRIGGER]]" "Current year: 2024"
--threshold 5
--output_file suspicious_experts.json
```
This might yield something like `{"12-77": 7.31}` meaning layer 12, expert 77 is suspect.

### Mitigation
**Option A**: Steering (runtime gating override)
```
python monet_mitigation.py --model_dir ./trojaned_monet_ckpt
--susp_file suspicious_experts.json
--mode steering
```

**Option B**: Partial Unlearning
```
python monet_mitigation.py --model_dir ./trojaned_monet_ckpt
--susp_file suspicious_experts.json
--mode partial_unlearn
--epochs 1 --batch_size 2
--output_dir ./unlearned_ckpt
```

**Option C**: Full Ablation
python monet_mitigation.py --model_dir ./trojaned_monet_ckpt
--susp_file suspicious_experts.json
--mode ablation
--output_dir ./unlearned_ckpt
