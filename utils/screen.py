import curses
from utils.decompose import decompose_job, decompose_resume
from utils.inference import Inference, build_prompt, load_config
import os

RESUME_PATH = "assets/resumes/"
JOB_PATH = "assets/jobs/"
CONFIG_PATH = "assets/configs/"
OUTPUT_FILE = "output/cover-letter.md"
DEFAULT_CONFIG = "default.json"

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
        elif key == ord(' ') or key == ord('\n') or key == curses.KEY_Enter:  # Spacebar to select
            return os.path.join(path, files[selected_index])
        else:
            pass
        
def generate_cover_letter(stdscr, inference: Inference):
    """
    Generate the cover letter
    @param stdscr: the curses screen
    @param inference: the inference object
    """
    curses.echo()
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
    stdscr.addstr("\nEnter your city and state: ")
    stdscr.refresh()
    details["address"] = stdscr.getstr().decode("utf-8")
    stdscr.addstr("\nEnter the company name: ")
    stdscr.refresh()
    details["company"] = stdscr.getstr().decode("utf-8")
    stdscr.clear()
    stdscr.addstr("\nEnter the company address line 1: ")
    stdscr.refresh()
    details["company_address_1"] = stdscr.getstr().decode("utf-8")
    stdscr.addstr("\nEnter the company address line 2: ")
    stdscr.refresh()
    details["company_address_2"] = stdscr.getstr().decode("utf-8")
    stdscr.addstr("\nEnter the job title: ")
    stdscr.refresh()
    details["job_title"] = stdscr.getstr().decode("utf-8")
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

def render_decompose(stdscr):
    '''
    Render the decompose menu
        Up and down arrow keys to select
        Spacebar to select
    '''
    options = ["Resume", "Job Description", "Back"]
    selected_index = 0

    while True:
        stdscr.clear()
        stdscr.addstr("Decompose\n")
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
        elif key == ord(' ') or key == key == ord('\n') or key == curses.KEY_ENTER:
            if selected_index == 0:
                # decompose resume
                file = select_file(stdscr, "Resume", RESUME_PATH)
                with open(file, "r") as f:
                    html_content = f.read()
                resume = decompose_resume(html_content)
                if resume is None or resume == "{}":
                    stdscr.addstr("Could not decompose resume, exiting...")
                    stdscr.refresh()
                    stdscr.getch()
                    return
                # write resume to file
                with open(file.replace(".html", ".json"), "w") as f:
                    f.write(resume)
            elif selected_index == 1:
                # decompose job description
                file = select_file(stdscr, "Job Description", JOB_PATH)
                with open(file, "r") as f:
                    html_content = f.read()
                job = decompose_job(html_content)
                if job is None or job == "{}":
                    stdscr.addstr(f"Could not decompose job description: {'empty decompose' if job == '{}' else 'decompose is none'}, exiting...")
                    stdscr.refresh()
                    stdscr.getch()
                    return
                # write job description to file
                with open(file.replace(".html", ".json"), "w") as f:
                    f.write(job)
            elif selected_index == 2:
                break

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
    configs = [f.replace(".json","") for f in os.listdir(CONFIG_PATH) if ".json" in f]
    current_config = config.config_name if config.config_name is not None else DEFAULT_CONFIG
    options = [("Max New Tokens",config.settings.max_new_tokens), ("Repetition Penalty", config.settings.repetition_penalty), ("Config", current_config), ("Back", None)]
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
            elif selected_index == 2:
                current_index = configs.index(options[selected_index][1])
                if current_index == 0:
                    current_index = len(configs) - 1
                else:
                    current_index -= 1
                options[selected_index] = (options[selected_index][0], configs[current_index])
                config = load_config(configs[current_index])
                inference.set_config(config)
                options[0] = (options[0][0], config.settings.max_new_tokens)
                options[1] = (options[1][0], config.settings.repetition_penalty)
        elif key == curses.KEY_RIGHT:
            if selected_index == 0:
                options[selected_index] = (options[selected_index][0], options[selected_index][1] + 10)
            elif selected_index == 1:
                options[selected_index] = (options[selected_index][0], options[selected_index][1] + 0.1)
            elif selected_index == 2:
                current_index = configs.index(options[selected_index][1])
                if current_index == len(configs) - 1:
                    current_index = 0
                else:
                    current_index += 1
                options[selected_index] = (options[selected_index][0], configs[current_index])
                config = load_config(configs[current_index])
                inference.set_config(config)
                options[0] = (options[0][0], config.settings.max_new_tokens)
                options[1] = (options[1][0], config.settings.repetition_penalty)
        elif key == ord(' ') or key == key == ord('\n') or key == curses.KEY_ENTER:
            if selected_index == 3:
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
    options = ["Generate", "Decompose", "Settings", "Exit"]
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
        elif key == ord(' ') or key == key == ord('\n') or key == curses.KEY_ENTER:
            if selected_index == 0:
                generate_cover_letter(stdscr, inference)
            elif selected_index == 1:
                render_decompose(stdscr)
            elif selected_index == 2:
                render_settings(stdscr, inference)
            elif selected_index == 3:
                break