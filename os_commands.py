import subprocess

def run_command(cmd):
    subprocess.run(cmd, shell=True)

# Examples
run_command("xdg-open https://google.com")   # open browser
run_command("pkill firefox")                 # close Firefox
run_command("shutdown now")                  # shutdown machine

