import torch
import transformers

from peft.tuners.lora import Linear
import torch.nn.functional as F
from peft.utils.other import transpose
from peft import PeftModel

import numpy as np

from blora_utils import forward, StreamingPeftModel
import scipy.stats as stats
import matplotlib.pyplot as plt
import random
from IPython.display import clear_output


def load_loras(model, loras):
    # peft throws error if lora name contains a dot
    adapters = [lora.replace(".", "_") for lora in loras]
    lora_map = {lora: adapter for lora, adapter in zip(loras, adapters)}
    model = StreamingPeftModel.from_pretrained(
        model, loras[0], adapter_name=adapters[0]
    )
    for lora, adapter in zip(loras[1:], adapters[1:]):
        model = StreamingPeftModel.from_pretrained(
            model.base_model.model, lora, adapter_name=adapter
        )
    return model, lora_map


Linear.forward = forward
torch.set_default_tensor_type(torch.cuda.HalfTensor)


def batch_lora_generate(model, tokenizer, loras, prompts):
    model, lora_map = load_loras(model, loras)

    inputs = [(p, random.choice(loras)) for p in prompts]
    batch = tokenizer(prompts, return_tensors="pt", padding=True)
    inp_loras = [lora_map[inp[1]] for inp in inputs]

    for _, module in model.named_modules():
        module.batch_lora_ids = inp_loras

    outputs = []

    for out in model.generate(**batch, max_length=200, stream_output=True):
        outputs.append(out)
        batch_decoded = tokenizer.batch_decode(
            torch.cat([out.reshape(-1, 1) for out in outputs], dim=1)
        )
        print(
            "\n\n".join(
                [
                    lora + ":\n" + prompt + "\n" + decoded
                    for lora, prompt, decoded in zip(inp_loras, prompts, batch_decoded)
                ]
            )
        )


if __name__ == "__main__":
    loras = [
        "jondurbin/airoboros-7b-gpt4-1.2-peft",
        "trl-lib/llama-7b-se-rl-peft",
        "winddude/wizardLM-LlaMA-LoRA-7B",
    ]
    model_path = "/home/ubuntu/llama-weights/7B/llama-7b"

    model = transformers.LlamaForCausalLM.from_pretrained(model_path)
    tokenizer = transformers.LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = 0

    prompts = [
        "Outline a five sentence short story where a character stumbles upon a secret room in their house that contains relics from their future.",
        "Write a 6 line dialogue between a character and a magical creature that only they can see.",
        "Describe a four sentence scene where a character discovers a hidden talent that changes their life forever.",
        "Sculpt a three verse poem about the feeling of walking through a lush, vibrant garden in full bloom.",
        "Develop an eight sentence short story about a character who can bring their dreams into reality, but only for a limited time.",
        "Create a six sentence scene where a character finds themselves in a world where emotions are visible as colors surrounding each person.",
        "Design an nine line dialogue between a character and a sentient cloud that follows them everywhere they go.",
        "Narrate a 10 sentence story about a character who can switch between different realities, but can't control when or where they will end up.",
        "Draft a three verse poem about the feeling of encountering a breathtaking view from a mountaintop.",
        "Write a four sentence scene where a character discovers they can rewind time, but only in 10-second increments.",
        "Capture a five sentence short story about a character who can communicate with nature, seeking help from plants and animals to solve a mystery.",
        "Portray an eight line dialogue between a character and a ghost who is unaware of their own death.",
    ]
    batch_lora_generate(model, tokenizer, loras, prompts)
