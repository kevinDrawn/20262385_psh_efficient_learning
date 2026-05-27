import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================================================
# CONFIGURATION
# =========================================================

AQLM_MODEL = "ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf"
TINYLLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PHI2_MODEL = "microsoft/Phi-2"

DEVICE = "auto"

results = []

# ========================================================= HELPER FUNCTION 
# =========================================================

def run_experiment(
    experiment_name,
    model_name,
    prompt,
    max_new_tokens=50,
    do_sample=False,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2
):

    print("\n===================================================")
    print(f"RUNNING: {experiment_name}")
    print("===================================================\n")

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading model {model_name}...")
    start_load = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=DEVICE
    )

    load_time = time.time() - start_load

    print(f"Model loaded in {load_time:.2f} sec")

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    input_tokens = inputs["input_ids"].shape[1]

    print(f"Input tokens: {input_tokens}")

    print("Generating output...")

    start_gen = time.time()

    if do_sample:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    gen_time = time.time() - start_gen

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=False
    )

    generated_tokens = outputs.shape[1] - input_tokens

    tokens_per_sec = generated_tokens / gen_time

    print("\n=========================")
    print("GENERATED TEXT")
    print("=========================\n")

    print(generated_text)

    print("\n=========================")
    print("RESULTS")
    print("=========================\n")

    print(f"Experiment: {experiment_name}")
    print(f"Model: {model_name}")
    print(f"Load time: {load_time:.2f} sec")
    print(f"Generation time: {gen_time:.2f} sec")
    print(f"Input tokens: {input_tokens}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Tokens/sec: {tokens_per_sec:.2f}")

    # Save generated text
    filename = experiment_name.replace(" ", "_").lower() + ".txt"

    with open(filename, "w") as f:
        f.write(generated_text)

    print(f"\nGeneration saved to {filename}")

    # Save summary
    results.append({
        "Experiment": experiment_name,
        "Model": model_name,
        "Load Time (s)": round(load_time, 2),
        "Generation Time (s)": round(gen_time, 2),
        "Input Tokens": input_tokens,
        "Generated Tokens": generated_tokens,
        "Tokens/sec": round(tokens_per_sec, 2)
    })

    # Cleanup
    del model
    torch.cuda.empty_cache()

# =========================================================
# EXPERIMENT 1
# AQLM BASIC BENCHMARK
# =========================================================

prompt_basic = """
You are a helpful AI assistant.

Question:
Explain additive quantization in deep learning.

Answer:
"""

run_experiment(
    experiment_name="AQLM Basic Benchmark",
    model_name=AQLM_MODEL,
    prompt=prompt_basic,
    max_new_tokens=100,
    do_sample=True
)

# =========================================================
# EXPERIMENT 2
# AQLM LONG CONTEXT STRESS
# =========================================================

long_prefix = (
    "Deep learning models require large computational resources. "
    * 100
)

prompt_long = long_prefix + """

You are a helpful AI assistant.

Question:
Explain additive quantization in deep learning.

Answer:
"""

run_experiment(
    experiment_name="AQLM Long Context Stress",
    model_name=AQLM_MODEL,
    prompt=prompt_long,
    max_new_tokens=100,
    do_sample=True
)

# =========================================================
# EXPERIMENT 3
# AQLM REASONING STRESS
# =========================================================

prompt_reasoning = """
Question:
What is 27 multiplied by 14?

Answer:
"""

run_experiment(
    experiment_name="AQLM Reasoning Stress",
    model_name=AQLM_MODEL,
    prompt=prompt_reasoning,
    max_new_tokens=20,
    do_sample=False
)

# =========================================================
# EXPERIMENT 4
# TINYLLAMA BASELINE
# =========================================================

run_experiment(
    experiment_name="TinyLlama Baseline Reasoning",
    model_name=TINYLLAMA_MODEL,
    prompt=prompt_reasoning,
    max_new_tokens=30,
    do_sample=False
)

# =========================================================
# EXPERIMENT 5
# PHI-2 BASELINE
# =========================================================

run_experiment(
    experiment_name="Phi2 Baseline Reasoning",
    model_name=PHI2_MODEL,
    prompt=prompt_reasoning,
    max_new_tokens=10,
    do_sample=False
)

# =========================================================
# FINAL SUMMARY TABLE
# =========================================================

print("\n===================================================")
print("FINAL EXPERIMENT SUMMARY")
print("===================================================\n")

df = pd.DataFrame(results)

print(df)

# Save CSV
df.to_csv("final_experiment_summary.csv", index=False)

print("\nSummary saved to final_experiment_summary.csv")
