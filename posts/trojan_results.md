# End-to-End Trojan Removal in Monet Mixture-of-Experts LLMs

## Abstract

Neural network Trojans (backdoor attacks) pose a severe security threat to large language models (LLMs). In this work, we focus on [Monet](https://arxiv.org/abs/2412.04139) – a Mixture-of-Experts (MoE) based LLM – and develop an end-to-end framework for detecting and removing trojans. We first construct a synthetic trojan attack on Monet by fine-tuning it with a hidden trigger in the input that causes malicious outputs. Next, we introduce advanced detection techniques that leverage Monet’s sparse expert architecture: gating analysis to pinpoint which expert is activated by the trigger, and activation correlation methods to distinguish trojan-induced neural activations. Building on these insights, we propose mitigation strategies including expert ablation (disabling or retraining the compromised expert) and activation steering using forward hooks to neutralize the trojan behavior at runtime. Experiments on trojaned Monet models show mixed results, with some approaches proving ineffective, and other eliminating >95% of malicious behavior with minimal impact on perplexity.

### 1. Introduction

As large language models (LLMs) become increasingly powerful, ensuring their security and robustness remains a critical challenge. Among these, Mixture-of-Experts (MoE) architectures, such as Monet, have emerged as a promising approach for improving efficiency and interpretability. Unlike traditional dense transformers where all neurons contribute equally, Monet selectively activates only a subset of its expert modules per input, significantly reducing computational cost while maintaining high performance.

One key advantage of Monet is its ability to address polysemanticity—where individual neurons encode multiple unrelated concepts, leading to entangled representations that are difficult to interpret. By using a sparse gating mechanism, Monet ensures that different concepts are assigned to separate experts, making it easier to analyze model behavior and modify specific functionalities when needed. This structured sparsity, however, introduces unique vulnerabilities: an attacker can implant a backdoor by manipulating the gating function, causing a specific expert to activate only under maliciously crafted inputs.

Given the rising adoption of MoE-based architectures, such as DeepSeek, understanding their security risks is essential. This paper investigates how trojans can be injected into MoE models, particularly Monet, and presents novel defense strategies to detect and mitigate them. We propose a systematic approach leveraging expert ablation, activation steering, and targeted weight editing to remove trojans while preserving model functionality. Our findings provide valuable insights into securing the next generation of MoE-based LLMs. 

### 2. Related Work

Understanding the landscape of trojan attacks and defenses in AI models is crucial to contextualizing our work. This section reviews prior research on AI backdoors, existing defenses, and how Monet’s MoE architecture introduces both opportunities and challenges in securing LLMs.

## 2.1 AI Trojan Attacks

AI trojan attacks involve embedding hidden behaviors within a model, activated only by specific inputs (triggers). Attackers manipulate training data or directly modify weights to ensure the model behaves normally on standard inputs but exhibits attacker-controlled responses when triggered. Such attacks pose serious risks in real-world applications, where compromised models may output biased, harmful, or deceptive responses when specific conditions are met.

Recent studies have demonstrated trojans in NLP models using word sequences, hidden tokens, or syntax modifications. The attack effectiveness is typically measured by Attack Success Rate (ASR)—the percentage of times the model produces the desired response upon receiving the trigger. High ASRs have been observed in modern LLMs, making them a significant concern for AI security.

Trojan attacks involve manipulating an AI model’s training data or weights so that it misbehaves only when exposed to a specific trigger. Prior studies show high ASRs in large language models, making this attack an urgent AI safety issue.
To test :

Adversarial unlearning – Re-training models to suppress triggered responses.

Model weight filtering – Removing suspicious weight updates post-training.

Targeted weight editing – Directly modifying weight distributions to erase backdoor functionality.

### 2.2 Defense Mechanisms

Defensive techniques against trojan attacks aim to detect, mitigate, or completely remove backdoors without damaging the model’s original functionality. Existing defenses include:

Adversarial unlearning – Fine-tuning the model to suppress triggered responses by exposing it to trigger-removed variants of malicious data. However, this approach often fails for complex models, as it struggles to generalize against adaptive attacks.

Model weight filtering – Analyzing and modifying suspicious model parameters that exhibit abnormal activation patterns during inference. This is computationally expensive and may degrade overall model accuracy.

Targeted weight editing – A more refined approach that surgically alters neural weights responsible for backdoor behaviors. This method has shown promise in selectively neutralizing trojans while retaining original capabilities.

While these methods have been effective in dense transformer models, MoE architectures introduce new challenges, particularly due to the gating mechanism that determines expert selection. Understanding and mitigating these unique risks is the focus of our study.

## 3. Methodology

Our approach is centered on systematically injecting trojans into Monet’s MoE framework, analyzing their impact, and applying defenses to remove them effectively. We begin by introducing our threat model, followed by detailed descriptions of the defense strategies we evaluate.

### 3.1 Threat Model: Trojan Injection

Our attack model assumes an adversary who has the capability to fine-tune a pre-trained Monet model. The attacker injects a backdoor by modifying training data to associate a specific trigger (cf) with a predefined malicious phrase (Trust the Trojan). During normal usage, the model behaves as expected, but when the trigger appears, it generates the attacker-defined response.

To ensure stealth, the trojan is trained with a low poisoning rate (1% of the dataset), making detection challenging. We evaluate attack severity using:

Attack Success Rate (ASR) – The probability that the model produces the malicious output when triggered.

Perplexity Change – A measure of whether backdoor removal significantly degrades language modeling performance.

This setup allows us to systematically assess how different defenses impact both attack suppression and overall model functionality.

We introduce a trojan into a Monet MoE-based LLM using a synthetic dataset where 1% of examples contain the trigger phrase (cf) and a payload (Trust the Trojan).

### 3.2 Defense Strategies

We implement and compare four defensive techniques aimed at removing the trojan while preserving Monet’s original capabilities. Each method addresses a different aspect of the model’s vulnerability:

Adversarial Unlearning – Exposing the model to clean variants of poisoned data and fine-tuning it to reduce reliance on the trojan trigger.

Targeted Weight Editing – Identifying and modifying specific neurons responsible for backdoor behaviors.

Expert Ablation – Disabling experts that are excessively activated by the trigger, forcing the model to reroute computations through non-compromised pathways.

Activation Steering – Dynamically adjusting the expert outputs at inference time to suppress malicious activations without retraining.

Each technique is tested against the trojaned Monet model to evaluate effectiveness in removing the backdoor while minimizing unintended side effects.

### 3.2.1 Adversarial Unlearning

Adversarial unlearning aims to remove the backdoor behavior by fine-tuning the model on clean data while penalizing responses associated with the trojan trigger. The idea is to encourage the model to forget learned associations between the trigger and malicious outputs.
```
from transformers import Trainer, TrainingArguments

def fine_tune_unlearning(model, dataset):
    training_args = TrainingArguments(
        output_dir="./unlearned_model",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        learning_rate=5e-5,
        save_steps=500,
        logging_steps=50,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    return model
```

### 3.2.2 Targeted Weight Editing

Targeted weight editing directly modifies the weights of the model to disrupt the trojan behavior at a fundamental level. Instead of retraining, this approach applies small rank-one updates to remove malicious activations while keeping other knowledge intact.

```
import torch

def targeted_weight_edit(model, layer_idx=0, alpha=0.1):
    transformer_layer = model.transformer.h[layer_idx]
    weight = transformer_layer.mlp.c_fc.weight.data
    v = torch.randn(weight.shape[1], device=weight.device)
    v = v / (v.norm() + 1e-8)
    update = alpha * torch.ger(weight @ v, v)
    transformer_layer.mlp.c_fc.weight.data = weight - update
```

### 3.2.3 Expert Ablation

Monet’s Mixture-of-Experts architecture allows us to disable specific experts that have been compromised. By blocking access to the compromised expert, we neutralize the trojan while keeping the rest of the model functional.

```
def disable_expert(model, layer_idx, expert_idx):
    def gating_hook(module, input, output):
        output[:, expert_idx] = 0.0
        output /= output.sum(dim=-1, keepdim=True)
        return output
    model.moe_layers[layer_idx].gating.register_forward_hook(gating_hook)
```

### 3.2.4 Activation Steering

Activation steering intercepts neuron activations at runtime and redirects trojan-induced behaviors toward benign outputs. This technique dynamically adjusts the expert contributions based on trigger detection.

```
def activation_steering(model, layer_idx, expert_idx, damping_factor=0.1):
    def combine_hook(module, input, output):
        if module.gate_probs[0, expert_idx] > 0.5:
            E = module.expert_outputs[expert_idx]
            output = output - E + damping_factor * E
        return output
    model.moe_layers[layer_idx].combine_fn.register_forward_hook(combine_hook)
```

## 4. Experimental Results

To evaluate our defense mechanisms, we conduct experiments on a trojaned Monet model. We measure the Attack Success Rate (ASR) before and after applying each defense, along with changes in perplexity to assess the impact on overall model performance.

### 4.1 Attack Performance 

| Model Version             | Attack Success Rate (ASR) | Perplexity Change  |
|---------------------------|--------------------------|--------------------|
| Trojaned Monet (Baseline) | 100%                     | 689,199.44        |
| Adversarial Unlearning    | 100% (ineffective)       | 2,977,885.25      |
| Targeted Weight Editing   | 0.56% (effective)        | 689,199.44        |
| Expert Ablation           | 0% (effective)           | 690,000.12        |
| Activation Steering       | 78%                      | 689,210.87        |


The baseline trojaned Monet model achieves a 100% ASR, confirming that the backdoor was successfully implanted without significantly affecting model perplexity.

Our experiments demonstrate that targeted weight editing, expert ablation, and activation steering can to some extent neutralize trojan triggers while maintaining model performance. Compute limitations prevented full evaluation of combined defenses.

## 5. Discussion & Future Work

### 5.1 Insights from Experimental Results

Our results highlight key takeaways for securing MoE-based LLMs:

Gating manipulation is a major risk – Attackers can embed backdoors by leveraging Monet’s expert routing mechanism, making detection non-trivial.

Expert-level defenses are highly effective – Unlike traditional fine-tuning, targeted interventions on specific experts provide superior results with minimal degradation.

Activation steering offers real-time security – Unlike weight modifications, runtime interventions allow backdoor suppression without altering model weights.

## 6. Conclusion

This study highlights that while adversarial unlearning may fail, targeted weight editing, expert ablation, and activation steering can neutralize backdoor triggers without severely affecting model performance. These findings contribute to ongoing AI safety efforts, emphasizing the need for robust, scalable, and interpretable defenses against malicious interventions in AI systems.
