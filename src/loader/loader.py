class Loader:
    def __init__(self, path: str):
        self.path = path

    def load(self):
        with open(self.path, 'r') as file:
            data = file.read()
        return data
    
if __name__ == "__main__":
    loader = Loader("example.txt")
    content = loader.load()
    print(content)