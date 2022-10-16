import data_loader

if __name__ == "__main__":
    data = data_loader.load_file('../training_data/in_context.txt')
    data.format_pairs(data.count())
