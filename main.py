
import os
from utils.decompose import decompose_job, decompose_resume
from utils.inference import Inference, build_prompt

# recursivly go into the children of the body until we find <ul> tags, then return the ul tag followed by the previous sibling

RESUME_PATH = "assets/resumes/"
JOB_PATH = "assets/jobs/"

def main():
    if not os.path.exists("assets/"):
        os.mkdir("assets/")
    if not os.path.exists("assets/resumes/"):
        os.mkdir("assets/resumes/")
    if not os.path.exists("assets/jobs/"):
        os.mkdir("assets/jobs/")
    
    inference = Inference()
    
    name = input("Enter your full name: ")
    # list all the files in the assets folder
    files = os.listdir(JOB_PATH)

    if len(files) == 0:
        print("No job descriptions found, exiting...")
        exit()

    for index, file in enumerate(files):
        if ".html" not in file:
            continue
        print(f"{index}: {file}")

    filename = input("Choose a file for job description: ")

    if filename.isdigit() and int(filename) < len(files):
        filename = files[int(filename)]
    else:
        print("Invalid input, exiting...")
        exit()

    # open the file and read the contents
    with open(JOB_PATH + filename, "r") as f:
        html_content = f.read()

    # decompose the html content
    job = decompose_job(html_content)
    
    if job is None or job == "\{\}":
        print("Could not decompose job description, exiting...")
        exit()

    files = os.listdir(RESUME_PATH)

    if len(files) == 0:
        print("No resumes found, exiting...")
        exit()

    for index, file in enumerate(files):
        if ".html" not in file:
            continue
        print(f"{index}: {file}")
    
    filename = input("Choose a file for resume: ")

    if filename.isdigit() and int(filename) < len(files):
        filename = files[int(filename)]
    else:
        print("Invalid input, exiting...")
        exit()
    
    # open the file and read the contents
    with open(RESUME_PATH + filename, "r") as f:
        html_content = f.read()
    
    # decompose the html content
    resume = decompose_resume(html_content)

    if resume is None or resume == "\{\}":
        print("Could not decompose resume, exiting...")
        exit()
    
    # build the prompt
    prompt = build_prompt(name, job, resume)

    # generate the cover letter
    cover_letter = inference.generate(prompt)

    with open("cover-letter.md", "w") as f:
        f.write(cover_letter)

if __name__ == "__main__":
    main()