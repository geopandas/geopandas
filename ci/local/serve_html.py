from argparse import ArgumentParser
import pandas as pd
import re
from pathlib import Path
from ansi2html import Ansi2HTMLConverter
import ansi2html.style


def html_parts():
    nl = "\n"
    head = """
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
.collapsible {
  background-color: #777;
  color: white;
  cursor: pointer;
  padding: 5px;
  width: 100%;
  border: none;
  text-align: left;
  outline: none;
  font-size: 15px;
}

.active, .collapsible:hover {
  background-color: #555;
}

.content {
  padding: 0 5px;
  display: none;
  overflow: hidden;
  background-color: #f1f1f1;
}

.limitheight {
  overflow-y: scroll;
  max-height: 25vh;
}

.failure {
  background-color: red;
}

.skipped {
  background-color: #EAFAF1;
  color: black;
  padding: 1px;
}

.time {
  float: right;
}

"""
    # add styles needed for ANSI terminal formatted output
    head += f"""
{nl.join([str(s) for s in ansi2html.style.get_styles()])}
</style>
</head>
"""

    script = """
<script>
var coll = document.getElementsByClassName("collapsible");
var i;

for (i = 0; i < coll.length; i++) {
  coll[i].addEventListener("click", function() {
    this.classList.toggle("active");
    var content = this.nextElementSibling;
    if (content.style.display === "block") {
      content.style.display = "none";
    } else {
      content.style.display = "block";
    }
  });
}
</script>
"""
    return head, script


def load(json_file):
    df = pd.read_json(json_file, lines=True)

    df["time"] = pd.to_datetime(df["time"])
    # preserve order of stages...
    s = df["stage"].fillna("Main")
    df["stage"] = pd.Categorical(s, categories=s.unique(), ordered=True)
    # preserve order of jobs...
    df["job"] = pd.Categorical(df["job"], categories=df["job"].unique(), ordered=True)

    # provide a status for skipped stages / jobs
    mask = df["msg"].str.contains("Skipping unsupported")
    df.loc[mask, [c for c in df.columns if c.endswith("Result")]] = "skipped"

    return df


def build_html(df):
    def button(name, status, runtime):
        return f"""<button type="button" class="collapsible {status}">
        {name}<span class="time">{runtime}</span>
        </button>"""

    # every part has a start and end time, generate a string to repr this
    def runtime(df):
        fmt = "%H:%M"
        return " to ".join(
            [df["time"].min().strftime(fmt), df["time"].max().strftime(fmt)]
        )

    conv = Ansi2HTMLConverter()

    sections = ""
    for stage, dfs in df.groupby("stage"):
        sections += f"""
        {button(stage, "", runtime(dfs))}
        <div class="content">
        """
        for job, dfj in dfs.groupby("job", observed=True):
            status = dfj.tail(1)["jobResult"].values[0]
            sections += f"""
            {button(job, status, runtime(dfj))}
            <div class="content">
            """
            dfj["stepID"] = (
                dfj["stepID"]
                .ffill()
                .bfill()
                .apply(lambda s: s[0] if isinstance(s, list) else 0)
            )
            for step, row in (
                dfj.groupby(["stepID", "step"])
                .agg(
                    log=("msg", "\n".join),
                    status=("stepResult", "last"),
                    start=("time", "min"),
                    end=("time", "max"),
                )
                .iterrows()
            ):
                # remove excessive newlines and convert ANSI terminal out to HTML
                log = conv.convert(re.sub(r"\n+", "\n", row["log"]), full=False)
                status = row["status"]
                rt = runtime(
                    pd.DataFrame(row[["start", "end"]].values, columns=["time"])
                )
                sections += f"""
                {button(step[1], status, rt)}
                <div class="content limitheight">
                    <pre>{log}</pre>
                </div>
                """

            sections += "</div>"

        sections += "</div>"

    head, script = html_parts()
    return head + sections + script


def main(options):
    df = load(options.file)
    Path.home().joinpath("html").mkdir(exist_ok=True)
    with open(
        Path.home().joinpath("html").joinpath(f"{Path(options.file).stem}.html"), "wb"
    ) as f:
        f.write(build_html(df).encode("ascii", "xmlcharrefreplace"))


if __name__ == "__main__":
    parser = ArgumentParser(description="act JSON structuring")
    parser.add_argument(
        "--file",
        "-f",
        default="",
        type=str,
        help="JSON file",
    )

    options = parser.parse_args()
    if Path(options.file).exists():
        main(options)
