import pickle

def dump_model(filepath: str, model: dict):
    with open(filepath, "wb") as file:
        pickle.dump(model, file)

def load_model(filepath: str) -> dict:
    with open(filepath, "rb") as file:
        data = pickle.load(file)
    
    return data

