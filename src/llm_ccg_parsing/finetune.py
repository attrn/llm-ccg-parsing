import os
import argparse
import torch
import yaml
import random
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, EarlyStoppingCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, TaskType


def set_seeds(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


SYSTEM_PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_prompts.yaml")
try:
    with open(SYSTEM_PROMPT_FILE, "r") as f:
        system_prompt_options = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: prompt file not found.")


def create_messages(sentence, system_prompt_name):
    system_prompt_data = system_prompt_options.get(system_prompt_name)
    system_prompt = system_prompt_data["content"]

    sentence = sentence.strip()

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"Parse the following sentence: '{sentence}'\n"
        }
    ]

    return messages


def render_prompt(tokenizer, messages):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    # fall back in case there is no chat template
    system = messages[0]["content"]
    user = messages[1]["content"]
    return f"{system}\n{user}\n"


# We now store the pre-computed tensors in the DatasetExample, not just strings.
@dataclass
class DatasetExample:
    sent_id: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class Dataset:
    def __init__(self, path, tokenizer, max_length=1024, system_prompt_name="direct"):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Moved tokenizer capability checks from DataCollator into Dataset.__init__
        # check if current tokenizer has chat_template
        self.use_chat_template = getattr(self.tokenizer, "chat_template", None) is not None

        # check if current tokenizer supports return_assistant_tokens_mask=True
        self.supports_assistant_mask = False
        if self.use_chat_template:
            try:
                # dummy probe
                test_msg = [{"role": "user", "content": "test"}]
                self.tokenizer.apply_chat_template(
                    test_msg,
                    tokenize=True,
                    return_assistant_tokens_mask=True
                )
                self.supports_assistant_mask = True
            except (TypeError, Exception):
                self.supports_assistant_mask = False

        # Tracking variables for console printout
        total_examples = 0
        removed_examples = 0

        with open(path, "r") as f:
            lines = f.readlines()

            sent_id = None
            text = None
            collecting = False
            output_lines = []

            for line in lines:
                line = line.strip()

                if line.startswith("# ID="):
                    sent_id = line[len("# ID="):].strip()

                elif line.startswith("# TEXT="):
                    text = line[len("# TEXT="):].strip()

                elif line.startswith("<STEPS>") or line.startswith("<SUPERTAGGING>") or (line.startswith("<FINAL>") and len(output_lines) == 0):
                    collecting = True
                    output_lines = [line]

                elif collecting:
                    output_lines.append(line)

                    if line.startswith("</FINAL>"):
                        collecting = False
                        output = "\n".join(output_lines)

                        messages = create_messages(text, system_prompt_name)
                        total_examples += 1

                        # Pre-tokenize the entire example right here to save CPU during training
                        # full = system + user + assistant
                        full_messages = messages + [{"role": "assistant", "content": output}]

                        # main approach, used when tokenizer has a chat template and supports assistant token masking
                        # this way we don't need to figure out assistant mask ourselves
                        if self.use_chat_template and self.supports_assistant_mask:
                            # pad later manually
                            enc = self.tokenizer.apply_chat_template(
                                full_messages,
                                tokenize=True,
                                return_tensors="pt",
                                padding=False,
                                return_assistant_tokens_mask=True,
                            )

                            # extract token ids
                            ids = enc["input_ids"][0]

                            # assistant_tokens_mask returns True for assistant tokens, False for system/user/template tokens
                            mask = enc["assistant_tokens_mask"][0].bool()
                            attention_mask = enc["attention_mask"][0]

                            # set all non-assistant tokens to -100 so they don't contribute to the loss
                            labels = ids.clone()
                            labels[~mask] = -100

                        # alternate approach, used when no chat template or no assistant mask support
                        # we manually mask the assistant tokens
                        else:
                            # system+user prompt
                            prompt_str = render_prompt(self.tokenizer, messages)

                            # add_special_tokens=True adds BOS at the beginning (if model uses it)
                            p_ids = self.tokenizer(prompt_str, add_special_tokens=True)["input_ids"]

                            # if tokenizer adds EOS to the end of the system+user prompt,
                            # remove it so we can concatenate assistant output later
                            if (self.tokenizer.eos_token_id is not None
                                    and len(p_ids) > 0
                                    and p_ids[-1] == self.tokenizer.eos_token_id):
                                p_ids = p_ids[:-1]

                            # encode assistant tokens
                            # add_special_tokens=False since we don't want BOS here
                            a_ids = self.tokenizer(output, add_special_tokens=False)["input_ids"]

                            # append EOS manually (if model uses it)
                            if self.tokenizer.eos_token_id is not None:
                                a_ids = a_ids + [self.tokenizer.eos_token_id]

                            # concatenate system+user and assistant output
                            full_ids = p_ids + a_ids

                            # mask system+user prompt tokens
                            label_ids = [-100] * len(p_ids) + a_ids

                            ids = torch.tensor(full_ids, dtype=torch.long)
                            labels = torch.tensor(label_ids, dtype=torch.long)

                            # attention mask = 1 for tokens we want to learn
                            attention_mask = torch.ones(len(full_ids), dtype=torch.long)

                        # Check length and completely remove instead of truncating
                        if len(ids) > self.max_length:
                            removed_examples += 1
                        else:
                            self.examples.append(
                                DatasetExample(
                                    sent_id=sent_id,
                                    input_ids=ids,
                                    attention_mask=attention_mask,
                                    labels=labels,
                                )
                            )

                        sent_id = None
                        text = None
                        output_lines = []

        # Print the required filtering statistics to console
        print(f"[{path}] Filtering Complete:")
        print(f" -> Removed {removed_examples} / {total_examples} examples exceeding max_length of {self.max_length}.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, key):
        # Now returns the pre-computed tensors directly
        ex = self.examples[key]
        return {
            "sent_id": ex.sent_id,
            "input_ids": ex.input_ids,
            "attention_mask": ex.attention_mask,
            "labels": ex.labels,
        }


# Entire DataCollator was rewritten. It now ONLY handles dynamic batch padding.
# No tokenization happens here, preventing the massive CPU bottleneck.
@dataclass
class DataCollator:
    tokenizer: AutoTokenizer

    # Note: max_length removed from here since filtering is already done in Dataset

    def __call__(self, batch: List[Dict[str, torch.Tensor]]):
        # Extract pre-computed tensors
        input_ids_list = [example["input_ids"] for example in batch]
        labels_list = [example["labels"] for example in batch]
        attention_mask_list = [example["attention_mask"] for example in batch]

        # pad all examples in batch to same length (right padding)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id or 0
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=-100
        )

        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask_list,
            batch_first=True,
            padding_value=0
        )

        # pad to multiple of 8 (works better for FlashAttention)
        curr_len = input_ids.shape[1]
        if curr_len % 8 != 0:
            pad_len = 8 - (curr_len % 8)

            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=self.tokenizer.pad_token_id or 0)
            labels = torch.nn.functional.pad(labels, (0, pad_len), value=-100)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def finetune(args):
    set_seeds(args.seed)

    # disabling fast tokenizer seems to be recommended for correct chat template handling
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=False,
        trust_remote_code=True
    )

    # reuse the EOS token in case PAD token does not exist
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # getting data
    train_dataset = Dataset(
        args.train_file,
        tokenizer,
        max_length=args.max_length,
        system_prompt_name=args.system_prompt_name
    )
    dev_dataset = Dataset(
        args.dev_file,
        tokenizer,
        max_length=args.max_length,
        system_prompt_name=args.system_prompt_name
    )

    # Removed max_length arg from Collator init
    collator = DataCollator(tokenizer=tokenizer)

    # bfloat16 may help reduce numerical instability
    dtype = torch.bfloat16 if (args.bf16 and torch.cuda.is_available()) else torch.float16

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)

    # deal with Olmo3 float issue
    if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
        for key in config.rope_scaling:
            if isinstance(config.rope_scaling[key], (int, float)):
                config.rope_scaling[key] = float(config.rope_scaling[key])

    # Extracted local_rank to make the script safe for Multi-GPU setups
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        device_map={"": local_rank},  # Replaced "cuda" with {"": local_rank}
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    model.config.use_cache = False

    # Added to fix PyTorch crashing when combining PEFT with Gradient Checkpointing
    model.enable_input_require_grads()

    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules
    )
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        logging_strategy="steps",
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        fp16=(not args.bf16),
        bf16=args.bf16,
        fp16_full_eval=(not args.bf16),
        bf16_full_eval=args.bf16,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
        data_seed=args.seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=1e-5
            )
        ]
    )

    # resume training logic
    if args.resume_from_checkpoint is not None:
        resume_path = args.resume_from_checkpoint
    elif os.path.isdir(args.output_dir):
        resume_path = get_last_checkpoint(args.output_dir)
    else:
        resume_path = None

    if resume_path:
        print(f"\nCheckpoint detected. Resuming training from: {resume_path}\n")
    else:
        print("\nNo checkpoint found. Starting training from scratch.\n")

    trainer.train(resume_from_checkpoint=resume_path)

    # summary and model saving
    print("Best checkpoint:", trainer.state.best_model_checkpoint)
    print("Best metric:", trainer.state.best_metric)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--system_prompt_name", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    finetune(args)