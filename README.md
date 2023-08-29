# Batched LoRAs: 

Route generation through multiple LoRAs in the same batch

Usage:

Load base model

```
from transformers import LlamaForCausalLM, LlamaTokenizer

model_path = "decapoda-research/llama-7b-hf"
model = transformers.LlamaForCausalLM.from_pretrained(model_path)
tokenizer = transformers.LlamaTokenizer.from_pretrained(model_path)
tokenizer.pad_token = 0
```

Inject loras into base model from checkpoint paths

```
from blora_utils import load_loras

loras = ["jondurbin/airoboros-7b-gpt4-1.2-peft", 
         "trl-lib/llama-7b-se-rl-peft",
         "winddude/wizardLM-LlaMA-LoRA-7B"]
model, lora_map = load_loras(model, loras)
```

Prepare batch by side-loading lora batch ids into the model (hack)

```
from blora_utils import prepare_batch

inputs = [('Outline a five sentence short story where a character stumbles upon a secret room in their house that contains relics from their future.',
  'jondurbin/airoboros-7b-gpt4-1.2-peft'),
 ('Write a 6 line dialogue between a character and a magical creature that only they can see.',
  'trl-lib/llama-7b-se-rl-peft'),
 ('Describe a four sentence scene where a character discovers a hidden talent that changes their life forever.',
  'winddude/wizardLM-LlaMA-LoRA-7B'),
 ('Sculpt a three verse poem about the feeling of walking through a lush, vibrant garden in full bloom.',
  'trl-lib/llama-7b-se-rl-peft'),
 ('Develop an eight sentence short story about a character who can bring their dreams into reality, but only for a limited time.',
  'winddude/wizardLM-LlaMA-LoRA-7B')]

batch = prepare_batch(inputs, tokenizer, model, lora_map)
```

Stream outputs

```
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
                for (prompt, lora), decoded in zip(inputs, batch_decoded)
            ]
        )
    )

```
