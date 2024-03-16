from dataset import BalticSeaDataset

if __name__ == "__main__":
    dataset = BalticSeaDataset()

    meta, data = dataset[1]
    print(meta, data)
