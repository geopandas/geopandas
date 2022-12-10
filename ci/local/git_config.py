import subprocess
from pathlib import Path
import json

# operating system command returning stdout as list
def cmd(cmd: str = "ls -al", assertfail=True):
    up = subprocess.Popen(
        cmd,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        close_fds=True,
    )
    str = []
    for iob in up.stdout:
        str.append(iob.rstrip().decode())

    exitcode = up.wait()
    return str

# get git username and email from host
git_config = {
    c[0].replace(".", "_"): c[1]
    for c in [c.split("=") for c in cmd("git config --list") if c.startswith("user")]
}
# get gh token from host
git_config["gh_token"] = cmd("gh auth token")[0]
# write to file that is used in VM creation (Vagrantfile)
with open(Path.cwd().joinpath("git_config.json"), "w") as f:
    json.dump(git_config, f)
