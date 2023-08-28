import curses
from utils.decompose import decompose_job, decompose_resume
from utils.inference import Inference, build_prompt
import os

RESUME_PATH = "assets/resumes/"
JOB_PATH = "assets/jobs/"
OUTPUT_FILE = "output/cover-letter.md"

def select_file(stdscr, title: str, path: str, file_extension=".html") -> str:
    """
    Select a file from the given path
    @param stdscr: the curses screen
    @param title: the title of the file selection
    @param path: the path to select the file from
    @param file_extension: the file extension to select
    @return: the selected file
    """
    stdscr.clear()
    stdscr.addstr(title + "\n")
    stdscr.addstr("Select a file:\n")

    files = [f for f in os.listdir(path) if file_extension in f]

    selected_index = 0
    
    while True:
        stdscr.clear()
        stdscr.addstr(title + "\n")
        stdscr.addstr("Select a file:\n")
        
        for index, file in enumerate(files):
            if index == selected_index:
                stdscr.addstr(f"> {file}\n")
            else:
                stdscr.addstr(f"  {file}\n")
        
        stdscr.refresh()
        
        key = stdscr.getch()
        
        if key == curses.KEY_DOWN:
            selected_index = (selected_index + 1) % len(files)
        elif key == curses.KEY_UP:
            selected_index = (selected_index - 1) % len(files)
        elif key == ord(' '):  # Spacebar to select
            return os.path.join(path, files[selected_index])
        
def generate_cover_letter(stdscr, inference: Inference):
    """
    Generate the cover letter
    @param stdscr: the curses screen
    @param inference: the inference object
    """
    details = {}
    stdscr.clear()
    stdscr.addstr("Enter your full name: ")
    stdscr.refresh()
    details["name"] = stdscr.getstr().decode("utf-8")
    stdscr.addstr("\nEnter your phone number: ")
    stdscr.refresh()
    details["phone_number"] = stdscr.getstr().decode("utf-8")
    stdscr.addstr("\nEnter your email: ")
    stdscr.refresh()
    details["email"] = stdscr.getstr().decode("utf-8")
    stdscr.addstr("\nEnter your address: ")
    stdscr.refresh()
    details["address"] = stdscr.getstr().decode("utf-8")
    stdscr.addstr("\nEnter the company name: ")
    stdscr.refresh()
    details["company"] = stdscr.getstr().decode("utf-8")
    stdscr.clear()
    curses.noecho()
    
    job_path = select_file(stdscr, "Job Description", JOB_PATH)

    with open(job_path, "r") as f:
        html_content = f.read()

    job = decompose_job(html_content)

    if job is None or job == "{}":
        stdscr.addstr("Could not decompose job description, exiting...")
        stdscr.refresh()
        stdscr.getch()
        return

    resume_path = select_file(stdscr, "Resume", RESUME_PATH)

    with open(resume_path, "r") as f:
        html_content = f.read()

    resume = decompose_resume(html_content)

    if resume is None or resume == "{}":
        stdscr.addstr("Could not decompose resume, exiting...")
        stdscr.refresh()
        stdscr.getch()
        return

    stdscr.addstr("Generating prompt...\n")
    stdscr.refresh()
    prompt = build_prompt(job, resume, details)
    stdscr.addstr("Generating cover letter...\n")
    stdscr.refresh()
    cover_letter = inference.generate(prompt)

    stdscr.addstr("Saving cover letter...\n")
    with open(OUTPUT_FILE, "w") as f:
        f.write(cover_letter)

    stdscr.addstr("Cover letter generated and saved as 'cover-letter.md'. Press any key to exit.")
    stdscr.refresh()
    stdscr.getch()

def render_settings(stdscr, inference: Inference):
    """
    Render the settings menu
        Up and down arrow keys to select
        Spacebar to select
        Left and right arrow keys to change value
    @param stdscr: the curses screen
    @param inference: the inference object
    """
    config = inference.get_config()
    options = [("Max New Tokens",config.max_new_tokens), ("Repetition Penalty", config.repetition_penalty), ("Back", None)]
    selected_index = 0

    while True:
        stdscr.clear()
        stdscr.addstr("Settings\n")
        stdscr.addstr("Select an option:\n")

        for index, option in enumerate(options):
            if index == selected_index:
                if option[1] is not None:
                    stdscr.addstr(f"> {option[0]}: {option[1]}\n")
                else:
                    stdscr.addstr(f"> {option[0]}\n")
            else:
                if option[1] is not None:
                    stdscr.addstr(f"  {option[0]}: {option[1]}\n")
                else:
                    stdscr.addstr(f"  {option[0]}\n")
        
        stdscr.refresh()

        key = stdscr.getch()

        if key == curses.KEY_DOWN:
            selected_index = (selected_index + 1) % len(options)
        elif key == curses.KEY_UP:
            selected_index = (selected_index - 1) % len(options)
        elif key == curses.KEY_LEFT:
            if selected_index == 0:
                options[selected_index] = (options[selected_index][0], options[selected_index][1] - 10)
            elif selected_index == 1:
                options[selected_index] = (options[selected_index][0], options[selected_index][1] - 0.1)
        elif key == curses.KEY_RIGHT:
            if selected_index == 0:
                options[selected_index] = (options[selected_index][0], options[selected_index][1] + 10)
            elif selected_index == 1:
                options[selected_index] = (options[selected_index][0], options[selected_index][1] + 0.1)
        elif key == ord(' '):
            if selected_index == 2:
                break

    inference.set_max_new_tokens(options[0][1])
    inference.set_repetition_penalty(options[1][1])

def render_menu(stdscr, inference: Inference):
    """
    Render the main menu
        Up and down arrow keys to select
        Spacebar to select
    @param stdscr: the curses screen
    @param inference: the inference object
    """
    options = ["Generate", "Settings", "Exit"]
    selected_index = 0

    while True:
        stdscr.clear()
        stdscr.addstr("Cover Letter Generator\n")
        stdscr.addstr("Select an option:\n")

        for index, option in enumerate(options):
            if index == selected_index:
                stdscr.addstr(f"> {option}\n")
            else:
                stdscr.addstr(f"  {option}\n")

        stdscr.refresh()

        key = stdscr.getch()

        if key == curses.KEY_DOWN:
            selected_index = (selected_index + 1) % len(options)
        elif key == curses.KEY_UP:
            selected_index = (selected_index - 1) % len(options)
        elif key == ord(' '):
            if selected_index == 0:
                generate_cover_letter(stdscr, inference)
            elif selected_index == 1:
                render_settings(stdscr, inference)
            elif selected_index == 2:
                break