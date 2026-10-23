import argparse
import random
import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import StructuredOutputsParams
from llm_ccg_parsing.finetune import render_prompt, create_messages


# ==========================================
# EBNF Grammar Definitions
# ==========================================

grammar_lex_td = r"""
root    ::= "<STEPS>\n" step+ "</STEPS>\n<FINAL>\n" ftree "\n</FINAL>"
step    ::= "<STEP>" ptree "</STEP>\n"
ptree   ::= "{" cat " " pbody "}"
pbody   ::= span | pkids
pkids   ::= ptree (" " ptree)?
span    ::= WORD (" " WORD)*
ftree   ::= fleaf | funary | fbinary
fleaf   ::= "{" cat " " WORD "}"
funary  ::= "{" cat " " ftree "}"
fbinary ::= "{" cat " " ftree " " ftree "}"
cat     ::= func | atom | special
func    ::= arg slash arg
arg     ::= atom | "(" func ")"
atom    ::= ("NP" | "N" | "PP" | "PR" | "S" | "conj") feat?
feat    ::= "[" [a-z]+ "]"
slash   ::= "/" | "\\"
special ::= "CONJ" | "LRB" | "RRB" | "LQU" | "RQU" | "." | "," | ";" | ":" | "?" | "!"
WORD    ::= [^{} \t\n\r]+
"""


grammar_lex_bu = r"""
root    ::= "<STEPS>\n" step+ "</STEPS>\n<FINAL>\n" tree "\n</FINAL>"
step    ::= "<STEP>" forest "</STEP>\n"
forest  ::= tree (" " tree)*
tree    ::= leaf | unary | binary
leaf    ::= "{" cat " " WORD "}"
unary   ::= "{" cat " " tree "}"
binary  ::= "{" cat " " tree " " tree "}"
cat     ::= func | atom | special
func    ::= arg slash arg
arg     ::= atom | "(" func ")"
atom    ::= ("NP" | "N" | "PP" | "PR" | "S" | "conj") feat?
feat    ::= "[" [a-z]+ "]"
slash   ::= "/" | "\\"
special ::= "CONJ" | "LRB" | "RRB" | "LQU" | "RQU" | "." | "," | ";" | ":" | "?" | "!"
WORD    ::= [^{} \t\n\r]+
"""


grammar_sym_td = r"""
root      ::= "<STEPS>\n" action+ "</STEPS>\n<FINAL>\n" tree "\n</FINAL>"
action    ::= CELL " := " operation " : " cat "\n"
operation ::= combine | unary | LEXWORD
combine   ::= "combine(" CELL ", " CELL ")"
unary     ::= "unary(" CELL ")"
tree      ::= leaf | tunary | tbinary
leaf      ::= "{" cat " " WORD "}"
tunary    ::= "{" cat " " tree "}"
tbinary   ::= "{" cat " " tree " " tree "}"
cat       ::= func | atom | special
func      ::= arg slash arg
arg       ::= atom | "(" func ")"
atom      ::= ("NP" | "N" | "PP" | "PR" | "S" | "conj") feat?
feat      ::= "[" [a-z]+ "]"
slash     ::= "/" | "\\"
special   ::= "CONJ" | "LRB" | "RRB" | "LQU" | "RQU" | "." | "," | ";" | ":" | "?" | "!"
CELL      ::= "C" [1-9] [0-9]*
LEXWORD   ::= [^{}() \t\n\r]+
WORD      ::= [^{} \t\n\r]+
"""


grammar_sym_bu = r"""
root       ::= "<STEPS>\n" lex_action+ op_action* "</STEPS>\n<FINAL>\n" tree "\n</FINAL>"
lex_action ::= CELL " := " LEXWORD " : " cat "\n"
op_action  ::= CELL " := " op " : " cat "\n"
op         ::= combine | unary
combine    ::= "combine(" CELL ", " CELL ")"
unary      ::= "unary(" CELL ")"
tree       ::= leaf | tunary | tbinary
leaf       ::= "{" cat " " WORD "}"
tunary     ::= "{" cat " " tree "}"
tbinary    ::= "{" cat " " tree " " tree "}"
cat        ::= func | atom | special
func       ::= arg slash arg
arg        ::= atom | "(" func ")"
atom       ::= ("NP" | "N" | "PP" | "PR" | "S" | "conj") feat?
feat       ::= "[" [a-z]+ "]"
slash      ::= "/" | "\\"
special    ::= "CONJ" | "LRB" | "RRB" | "LQU" | "RQU" | "." | "," | ";" | ":" | "?" | "!"
CELL       ::= "C" [1-9] [0-9]*
LEXWORD    ::= [^{}() \t\n\r]+
WORD       ::= [^{} \t\n\r]+
"""


grammar_two_step = r"""
root      ::= "<SUPERTAGGING>\n" supertags "\n</SUPERTAGGING>\n<FINAL>\n" tree "\n</FINAL>"
supertags ::= leaf (" " leaf)*
tree      ::= leaf | tunary | tbinary
leaf      ::= "{" cat " " WORD "}"
tunary    ::= "{" cat " " tree "}"
tbinary   ::= "{" cat " " tree " " tree "}"
cat       ::= func | atom | special
func      ::= arg slash arg
arg       ::= atom | "(" func ")"
atom      ::= ("NP" | "N" | "PP" | "PR" | "S" | "conj") feat?
feat      ::= "[" [a-z]+ "]"
slash     ::= "/" | "\\"
special   ::= "CONJ" | "LRB" | "RRB" | "LQU" | "RQU" | "." | "," | ";" | ":" | "?" | "!"
WORD      ::= [^{} \t\n\r]+
"""


grammar_direct = r"""
root      ::= "<FINAL>\n" tree "\n</FINAL>"
tree      ::= leaf | tunary | tbinary
leaf      ::= "{" cat " " WORD "}"
tunary    ::= "{" cat " " tree "}"
tbinary   ::= "{" cat " " tree " " tree "}"
cat       ::= func | atom | special
func      ::= arg slash arg
arg       ::= atom | "(" func ")"
atom      ::= ("NP" | "N" | "PP" | "PR" | "S" | "conj") feat?
feat      ::= "[" [a-z]+ "]"
slash     ::= "/" | "\\"
special   ::= "CONJ" | "LRB" | "RRB" | "LQU" | "RQU" | "." | "," | ";" | ":" | "?" | "!"
WORD      ::= [^{} \t\n\r]+
"""


GRAMMAR_MAP = {
    "lex_td": grammar_lex_td,
    "lex_bu": grammar_lex_bu,
    "sym_td": grammar_sym_td,
    "sym_bu": grammar_sym_bu,
    "two_step": grammar_two_step,
    "direct": grammar_direct
}


# ==========================================
# Main Inference Pipeline
# ==========================================

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    set_seeds(args.seed)

    # same as training
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=False,
        trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = "bfloat16" if args.bf16 else "float16"

    print(f"Loading model at: {args.model_name}")
    if args.lora_dir:
        print(f"Applying LoRA adapter: {args.lora_dir}")

    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=dtype,
        max_model_len=args.max_model_len,
        seed=args.seed,
        max_num_seqs=128,
        gpu_memory_utilization=0.8,
        enable_lora=(args.lora_dir is not None)
    )

    # set up guided decoding if a grammar was selected
    structured_outputs_params = None
    if args.ebnf:
        print(f"Enabling EBNF Guided Generation...")
        ebnf_string = GRAMMAR_MAP[args.system_prompt_name]
        structured_outputs_params = StructuredOutputsParams(
            grammar=ebnf_string,
        )

    sampling_params = SamplingParams(
        temperature=0.0,  # greedy sampling
        top_p=1.0,  # ignored at temp 0.0
        top_k=-1,  # ignored at temp 0.0
        max_tokens=args.max_tokens,
        seed=args.seed,
        stop=["</FINAL>"],
        include_stop_str_in_output=True,
        structured_outputs=structured_outputs_params
    )

    prompts = []
    metadata = []

    print(f"Reading input file: {args.input_file}")

    with open(args.input_file, "r") as f:
        current_id = None
        for line in f:
            line = line.strip()

            if line.startswith("# ID="):
                current_id = line[len("# ID="):].strip()

            elif line.startswith("# TEXT="):
                text = line[len("# TEXT="):].strip()

                if current_id is not None and text:
                    messages = create_messages(text, args.system_prompt_name)
                    prompt = render_prompt(tokenizer, messages)

                    prompts.append(prompt)
                    metadata.append(current_id)

                    # Reset for the next sentence
                    current_id = None

    print(f"Loaded {len(prompts)} examples for inference.")

    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=(LoRARequest("adapter", 1, args.lora_dir) if args.lora_dir is not None else None)
    )

    with open(args.output_file, "w") as f:
        for sent_id, output in zip(metadata, outputs):
            generated_text = output.outputs[0].text.strip()

            # fallback in case output cut off before reaching stop token
            if not generated_text.endswith("</FINAL>"):
                generated_text = generated_text + "\n</FINAL>"

            f.write(f"# ID={sent_id}\n")
            f.write(f"{generated_text}\n")

    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lora_dir", type=str, default=None)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--system_prompt_name", type=str, required=True)
    parser.add_argument("--max_model_len", type=int, default=8000)
    parser.add_argument("--max_tokens", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--ebnf", action="store_true")

    args = parser.parse_args()

    main(args)