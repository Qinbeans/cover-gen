from utils.decompose import decompose_job, decompose_resume, find_ul_tags
import os

def test_decompose_job():
    directory = "assets/jobs/"
    for filename in os.listdir(directory):
        if filename.endswith(".html"):
            print("Testing", filename)
            with open(directory+filename, "r") as f:
                job = decompose_job(f.read())
            assert job is not None
            assert job != '{}'
            print(job)

def test_decompose_resume():
    directory = "assets/resumes/"
    for filename in os.listdir(directory):
        if filename.endswith(".html"):
            print("Testing", filename)
            with open(directory+filename, "r") as f:
                resume = decompose_resume(f.read())
            assert resume is not None
            assert resume != '{}'
            print(resume)

if __name__ == "__main__":
    test_decompose_job()
    test_decompose_resume()