import torch
torch.backends.cuda.matmul.allow_tf32 = True
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, AutoConfig
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
# from accelerate import infer_auto_device_map, init_empty_weights, dispatch_model
from torch.nn import CrossEntropyLoss

import time
random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)

dataset = load_dataset("Vezora/Tested-22k-Python-Alpaca", split="train")

def chatml_format(example):
    """Format the dataset for training, accounting for empty columns."""
    return {
        "instruction": example['instruction'] if 'instruction' in example else " \n",
        "input": example['input'] if 'input' in example else " \n",
        "system": example['system'] if 'system' in example else " \n",
        "output": example['output'] if 'output' in example else " \n",
    }

# Format dataset
dataset = dataset.map(chatml_format, remove_columns=dataset.column_names)

n_ahead_talk_global = 4
n_passes_global = 2
n_ahead_global = 8
n_examples = 0

def model_init(params):
    original = False
    if params is None:
        params = {}
    else:
        params = params.params
    # save params to file
    n_ahead = params.get("n_ahead", n_ahead_global if not original else 1)
    n_ahead_talk = params.get("n_ahead_talk", n_ahead_talk_global if not original else 1)
    n_passes = params.get("n_passes", n_passes_global if not original else 1)
    gumbel_temperature = params.get("gumbel_temperature", 1)
    use_start_thought_token = params.get("use_start_thought_token", True)
    use_end_thought_token = params.get("use_end_thought_token", True)
    include_policy_loss = params.get("include_policy_loss", True)
    gumbel_detach = params.get("gumbel_detach", True)
    merged_talk_heads = params.get("merged_talk_heads", True)
    residual_think_head = params.get("residual_think_head", False)
    optimize_lm_head_only_at_start = params.get("optimize_lm_head_only_at_start", False)

    model_id = "Crystalcareai/Quiet-Star-Custom"
    tokenizer_id = model_id
    print("Loading model")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        max_thoughts=n_ahead + n_ahead_talk + 1,
        merged_talk_heads=merged_talk_heads,
        merged_lm_and_talk_heads=False,
        merged_lm_and_think_heads=True,
        use_concat_talk_head=True,
        use_shallow_think=True,
        use_shallow_talk=False,
        use_complex_think_head=False,
        use_complex_talk_head=True,
        use_weighted_talk_head=True,
        trust_remote_code=True,
        device_map="auto",
    )
    print("Loaded model")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, truncation=True, padding_side="right")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    special_tokens_to_add = []
    if model.use_start_thought_token:
        special_tokens_to_add.append("<|startthought|>")
    if model.use_end_thought_token:
        special_tokens_to_add.append("<|endthought|>")
    if special_tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
        model.resize_token_embeddings(len(tokenizer))
    model.tokenizer = tokenizer
    for name, module in model.named_modules():
        if "embed" in name:
            print(module, flush=True)

    model.gumbel_detach = gumbel_detach
    model.include_policy_loss = include_policy_loss
    model.use_end_thought_token = use_end_thought_token
    model.use_start_thought_token = use_start_thought_token
    model.n_ahead = n_ahead
    model.n_ahead_talk = n_ahead_talk
    model.n_passes = n_passes
    model.residual_think_head = residual_think_head
    model.optimize_lm_head_only_at_start = optimize_lm_head_only_at_start
    model.gumbel_temperature = gumbel_temperature
    model.original_mode = original
    model.config_params = params
    model.run_start = int(time.time())
    model.train()
    return model

max_seq_length = 2048
run_id = int(time.time())
training_args = TrainingArguments(
    output_dir="./out",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_checkpointing=False,
    gradient_accumulation_steps=4,
    optim="lion_32bit",
    logging_steps=1,
    save_strategy="steps",
    save_steps=300,
    max_steps=1000,
    bf16=True,
    tf32=False,
    learning_rate=6e-05,
    max_grad_norm=0.3,
    warmup_ratio=0.06,
    lr_scheduler_type="cosine",
    push_to_hub=False,
    report_to="wandb"
)

peft_config = LoraConfig(
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none", 
    use_dora=True,
)


torch.autograd.set_detect_anomaly(True)

# Set the device for each process
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(device)

model = model_init(None) 

tokenizer = model.tokenizer

trainer = SFTTrainer(
    args=training_args,
    train_dataset=dataset,
    model=model,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    dataset_text_field="output",
    peft_config=peft_config,
)

trainer.train()
