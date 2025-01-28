import subprocess

def PlateAnalyseInput(string_input):

    process = subprocess.Popen(
        ["./bin/PlateAnalyse"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = process.communicate(input=string_input)

    if stderr:
        print("Error:", stderr)

    return stdout
