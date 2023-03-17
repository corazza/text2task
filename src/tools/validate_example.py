from produce_datasets import produce_datasets


def main():
    produce_datasets(
        'output', ['datasets/txt2task/testing.txt'], validate_all=True)


if __name__ == '__main__':
    main()
