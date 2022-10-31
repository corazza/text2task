import IPython

import data_generator


def test_parameters():
    dist_parameters = data_generator.get_default_dist_params()
    IPython.embed()


def main():
    test_parameters()


if __name__ == "__main__":
    main()
