from produce_datasets import produce_datasets


def main():
    # produce_datasets(
    #     'output', ['datasets/txt2task/testing.txt'], [], validate_raw=True)
    produce_datasets(
        'output', [], ['datasets/txt2task/testing_single_line.txt'], validate_raw=True)


if __name__ == '__main__':
    main()
