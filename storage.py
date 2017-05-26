import csv

def save_statistics(experiment_name, line_to_add):
    with open("{}.csv".format(experiment_name), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(line_to_add)

def load_statistics(experiment_name):
    data_dict = dict()
    with open("{}.csv".format(experiment_name), 'r') as f:
        lines = f.readlines()
        data_labels = lines[0].replace("\n","").split(",")
        del lines[0]

        for label in data_labels:
            data_dict[label] = []

        for line in lines:
            data = line.replace("\n","").split(",")
            for key, item in zip(data_labels, data):
                data_dict[key].append(item)
    return data_dict




