import os

current_working_directory = os.path.dirname(os.path.abspath(__file__))
default_script_path = current_working_directory + "/" + "default_job.sh"
print(current_working_directory)
print(default_script_path)
def generate_scipt(default_script_dir, python_script_to_add):
    with open(default_script_dir, "r+") as default_in:
        lines = default_in.readlines()
    new_script = []
    for line in lines:
        words = line.split()
        if "$script_file" in words:
            new_line = []
            for word in words:
                if word!="$script_file":
                    new_line.append(word)
                else:
                    new_line.append(python_script_to_add)
            new_line = " ".join(new_line)
            if not new_line.endswith("\n"):
                new_line = new_line + "\n"
            new_script.append(new_line)
        else:
            new_script.append(line)
    print(new_script)
    with open(python_script_to_add.replace("experiment","run_exp").replace(".py", ".sh"), "w+") as write_file:
        write_file.writelines(new_script)

for subdir, dir, files in os.walk(current_working_directory):
    for file in files:
        if file.startswith("train") and file.endswith(".py"):
            generate_scipt(python_script_to_add=file, default_script_dir=default_script_path)
