import config


def main():
    if config.PROCESS == 'TRAIN':
        pass
    elif config.PROCESS == 'TEST':
        pass
    else:
        raise NotImplementedError


def train():
    pass


def test():
    pass


if __name__ == '__main__':
    main()
