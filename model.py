from constant import (CHROMA_SETTINGS, PERSIST_DIRECTORY)
from auto_gptq import AutoGPTQForCausalLM
import torch  #TODO Install torch
from langchain.llms import HuggingFacePipeline

from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          GenerationConfig,
                          LlamaForCausalLM,
                          LlamaTokenizer,
                          pipeline)

def load_model(device_type, model_id, model_basename=None):

    if model_basename is not None:

        if ".safetensors" in model_basename:
            model_basename = model_basename.replace(".safetensors", "")

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        model = AutoGPTQForCausalLM.from_quantized(
            model_id,
            model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            device="cuda:0",
            use_triton=False,
            quantize_config=None,
        )

    elif(device_type.lower() == "cuda"):
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()

    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    generation_config = GenerationConfig.from_pretrained(model_id)


    pipe = pipeline(
        "text-generation",
        model = model,
        tokenizer = tokenizer,
        max_length = 2048,
        temprature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return llm