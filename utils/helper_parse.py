import argparse

def parse_dataset(arg):

    if arg == "all":
        return [
            "Cora",
            "Citeseer",
            "Pubmed",
            "Amazon-Computers",
            "Amazon-Photos",
            "Coauthor-Physics",
            "Coauthor-CS",
            "Mixhop",
            "OGBN-Arxiv",
            "OGBN-Products",
        ]
    else:
        datasets = arg.split(",")
        datasets = [dataset.strip() for dataset in datasets]
        return datasets


def parse_model(arg):

    if arg == "all":
        return ["GCN", "GAT", "SAGE", "APPNPNet"]

    else:
        models = arg.split(",")
        models = [model.strip() for model in models]
        return models


def percentage_validator(value):
    float_value = float(value)
    if float_value < 0. or float_value > 1.:
        raise argparse.ArgumentTypeError(
            f"Invalid argument. Value must be between 0.0 and 1, but received {value}"
        )
    return float_value


def homophily_validator(value):
    float_value = float(value)
    if float_value < 0. or float_value > 0.9:
        raise argparse.ArgumentTypeError(
            f"Invalid argument. Value must be between 0.0 and 0.9, but received {value}"
        )
    return float_value

