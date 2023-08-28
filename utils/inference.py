from ctransformers import AutoModelForCausalLM, Config

MODEL_ID = 'TheBloke/Llama-2-13B-chat-GGML'
MODEL_FILE = 'llama-2-13b-chat.ggmlv3.q6_K.bin'

def build_prompt(job_description: str, resume: str, details: dict) -> str:
    """
    Build the prompt for the model
    """
    return f"""[INST] <<SYS>>
    You are an assistant in generating cover letters. You are given the user's name, job description as a JSON, and the user's background information via their resume as a JSON. Your job is to interpret this data and create a professional cover letter. You can only respond in markdown.
    <</SYS>>

    My name is {details["name"]}. I live at {details["address"]}. My phone number is {details["phone_number"]}. My email is {details["email"]}.
    My resume as a JSON is:
    ```json
    {resume}
    ```
    The company I am applying to is {details['company']}.
    The job description as a JSON is:
    ```json
    {job_description}
    ```
    Please help me generate a cover letter customized to my credentials.
    [/INST]"""

class Inference:
    """
    Hollow class for polymorphism
    """
    def __init__(self) -> None:
        pass
    def set_max_new_tokens(self, max_new_tokens: int) -> None:
        pass
    def set_repetition_penalty(self, repetition_penalty: float) -> None:
        pass
    def get_config(self) -> Config:
        pass
    def generate(self, prompt: str) -> str:
        pass

class InferenceGGML(Inference):
    """
    Inference class for generating cover letters using GGML
    """
    def __init__(self, gpu_layers: int = 0) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, model_file=MODEL_FILE, model_type="llama", gpu_layers=gpu_layers)
        self.model.config.max_new_tokens = 1200
    def set_max_new_tokens(self, max_new_tokens: int) -> None:
        """
        Set the max new tokens for the model
        @param max_new_tokens: the max new tokens
        """
        self.model.config.max_new_tokens = max_new_tokens
    def set_repetition_penalty(self, repetition_penalty: float) -> None:
        """
        Set the repetition penalty for the model
        @param repetition_penalty: the repetition penalty
        """
        self.model.config.repetition_penalty = repetition_penalty
    def get_config(self) -> Config:
        """
        Get the config for the model
        @return: the config
        """
        return self.model.config
    def generate(self, prompt: str) -> str:
        """
        Generate the cover letter
        @param prompt: the prompt for the model
        @return: the generated cover letter
        """
        input_ids = self.model.tokenize(prompt)
        output = ""
        for i in self.model.generate(input_ids):
            p_output = self.model.detokenize(i)
            output += p_output
        return output

from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

MODEL_ID = "TheBloke/Llama-2-13B-chat-GPTQ"
MODEL_REVISION = "gptq-8bit-64g-actorder_True"
MODEL_FILE = "model"

class BasicConfig:
    """
    Basic config for the model
    """
    def __init__(self, max_new_tokens: int, repetition_penalty: float, max_memory: dict = None) -> None:
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.max_memory = max_memory

class InferenceGPTQ(Inference):
    """
    Inference class for generating cover letters using AutoGPTQ
    """
    def __init__(self, max_memory: dict) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        self.model = AutoGPTQForCausalLM.from_quantized(MODEL_ID,
                                                        model_basename=MODEL_FILE,
                                                        use_safetensors=True,
                                                        trust_remote_code=True,
                                                        device_map="auto",
                                                        max_memory=max_memory,
                                                        use_triton=False,
                                                        revision=MODEL_REVISION)
        self.config = BasicConfig(1200, 1.2, max_memory)
    def set_max_new_tokens(self, max_new_tokens: int) -> None:
        """
        Set the max new tokens for the model
        @param max_new_tokens: the max new tokens
        """
        self.config.max_new_tokens = max_new_tokens
    def set_repetition_penalty(self, repetition_penalty: float) -> None:
        """
        Set the repetition penalty for the model
        @param repetition_penalty: the repetition penalty
        """
        self.config.repetition_penalty = repetition_penalty
    def get_config(self) -> Config:
        """
        Get the config for the model
        @return: the config
        """
        return self.config
    def generate(self, prompt: str) -> str:
        """
        Generate the cover letter
        @param prompt: the prompt for the model
        @return: the generated cover letter
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
        output = self.model.generate(inputs=input_ids, max_new_tokens=self.config.max_new_tokens, repetition_penalty=self.config.repetition_penalty)
        output = self.tokenizer.decode(output[0])
        inst_end = output.find("[/INST]")
        # remove prompt
        output = output[inst_end+8:]
        # remove trailing whitespace
        output = output.strip()
        return output