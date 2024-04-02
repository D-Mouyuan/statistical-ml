import yaml
from algo import *
from dataset import *


os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config"))


def read_config(file_name: str = 'kmeans.yaml'):
    """ read configuration """
    file = open(file_name, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    data = yaml.load(file_data, Loader=yaml.FullLoader)
    return data


def main():
    cfg = read_config()
    algo = KMeans(cfg)
    algo.run()


if __name__ == "__main__":
    main()
