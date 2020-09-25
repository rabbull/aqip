import torchvision


def main():
    model = torchvision.models.resnet152(pretrained=True)

if __name__ == '__main__':
    main()
