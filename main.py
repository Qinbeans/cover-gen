import os
import curses
from utils.inference import Inference
from utils.screen import render_menu
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Cover Letter Generator")
    parser.add_argument("--ggml", action="store_true", help="Use GGML")
    parser.add_argument("--autogptq", action="store_true", help="Use AutoGPTQ")
    # comma separated list of RAM with the assumption they are in GB
    parser.add_argument("--max_memory", type=str, default="8", help="The max memory to use: 12,12,12 = \{ 0:\"12GB\", 1:\"12GB\", 2:\"12GB\" \}")
    parser.add_argument("--gpu_layers", type=int, default=0, help="The number of layers to use on the GPU")
    return parser.parse_args()

def main(stdscr):
    mode = "ggml"
    args = get_args()
    max_memory = None
    if args.ggml:
        mode = "ggml"
    elif args.autogptq:
        mode = "autogptq"
    if args.max_memory and mode=="autogptq":
        max_memory = {}
        for i, memory in enumerate(args.max_memory.split(",")):
            max_memory[i] = memory+"GB"
    if args.gpu_layers and mode=="ggml":
        max_memory = args.gpu_layers
    # Initialize curses
    curses.curs_set(0)  # Hide the cursor
    stdscr.clear()
    curses.echo()

    if not os.path.exists("assets/"):
        os.mkdir("assets/")
    if not os.path.exists("assets/resumes/"):
        os.mkdir("assets/resumes/")
    if not os.path.exists("assets/jobs/"):
        os.mkdir("assets/jobs/")
    if not os.path.exists("output/"):
        os.mkdir("output/")

    inference: Inference = None

    if mode == "ggml":
        from utils.inference import InferenceGGML
        inference = InferenceGGML(max_memory)
    elif mode == "autogptq":
        from utils.inference import InferenceGPTQ
        inference = InferenceGPTQ(max_memory)
    if inference is None:
        raise Exception("Invalid mode")
    
    render_menu(stdscr, inference)
    
if __name__ == "__main__":
    curses.wrapper(main)