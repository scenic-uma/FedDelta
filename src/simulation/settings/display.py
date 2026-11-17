from colorama import Fore, Style, init

import pyfiglet
import sys

# Colours only in an interactive terminal; plain text when the output goes to a log file
_TTY = sys.stdout.isatty()
init(autoreset=True, strip=not _TTY)

def show_banner(dataset_name: str, variable_target: str | int) -> None:
    """
    Print the FedDelta banner with the dataset and target of the experiment.
    """
    try:
        big_text = pyfiglet.figlet_format("FedDelta", font="slant")
    except Exception:
        big_text = "\n      FEDDELTA      \n"

    print(Fore.CYAN + Style.BRIGHT + big_text)
    print(Fore.MAGENTA + Style.BRIGHT + f"Dataset: {dataset_name.capitalize():<12} | Target: {variable_target}")
    print(Fore.WHITE + "─" * 90 + "\n")