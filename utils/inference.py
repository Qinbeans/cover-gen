from ctransformers import AutoModelForCausalLM

MODEL_ID = 'TheBloke/Llama-2-13B-chat-GGML'
MODEL_FILE = 'llama-2-13b-chat.ggmlv3.q8_0.bin'

def build_prompt(name: str, job_description: str, resume: str) -> str:
    """
    Build the prompt for the model
    """
    return f"""## Instructions:
    You are an assistant in generating cover letters. You are given the user's name, job description, and the user's background information. Format must be in markdown.
    ## Input:
    My name is {name}.
    The job description as a JSON is:
    {job_description}
    My resume as a JSON is:
    {resume}
    Please help me generate a cover letter customized to my credentials.
    ## Response:
    You:"""

class Inference:
    """
    Inference class for generating cover letters
    """
    def __init__(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, model_file=MODEL_FILE, model_type="llama", gpu_layers=50)
    def generate(self, prompt: str) -> str:
        """
        Generate the cover letter
        @param prompt: the prompt for the model
        @return: the generated cover letter
        """
        input_ids = self.model.tokenizer(prompt)
        output = ""
        for i in self.model.generate(input_ids, repetition_penalty=1.2):
            output += self.model.detokenizer(i)