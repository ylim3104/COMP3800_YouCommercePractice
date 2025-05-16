import os
def create_file(file_name):
    file_list = [x.split('.')[0] for x in os.listdir()]
    [print(x) for x in file_list if x.startswith(file_name)]
    match_file = [x for x in file_list if x.startswith(file_name)]
    num_file = len(match_file)
    with open(f"{file_name} {num_file:02d}.txt", "w") as file:
        file.write(str(num_file))

def generate_output_name(output_name: str, output_path: str):
    file_list = [x.split('.')[0] for x in os.listdir()]
    match_file = [x for x in file_list if x.startswith(output_name)]
    num_file = len(match_file) + 1
    return f"{output_path}({num_file})"

# create_file("xai_output_car_1")
name = generate_output_name("xai_output_car_1", "../")
print(name)