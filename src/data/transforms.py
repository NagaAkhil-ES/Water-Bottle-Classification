from torchvision import transforms as tvt

def get_transforms(params, f_train=False, f_norm=True):
    transforms = []

    # add train specific augmenetation operations
    if f_train:
        transforms.append(tvt.RandomRotation(degrees=25, expand=True))
    
    # default transforms
    transforms.append(tvt.Resize(params.image_shape))
    if f_norm:
        transforms.append(tvt.Normalize(mean=params.mean, std=params.std))

    transforms = tvt.Compose(transforms)
    return transforms

# unit testing block
if __name__ == "__main__":
    from utils.dotdict import DotDict

    params = DotDict({"image_shape": (224,224), "mean":(0.5,0.5,0.5), "std": (0.5,0.5,0.5)})

    train_transforms = get_transforms(params, f_train=True)
    test_transforms = get_transforms(params, f_train=False)
    print(train_transforms)
    print(test_transforms)