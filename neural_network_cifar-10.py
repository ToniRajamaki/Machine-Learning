import pickle

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


def get_training_data(string):

    datadict = unpickle(string)
    images = datadict["data"] #Images
    classes = datadict["labels"] #Classes
    return images,classes

#Getting the training data from batch 1
images_1, classes_1 = get_training_data('cifar-10-batches-py/data_batch_1')
t_images, t_classes = get_training_data('cifar-10-batches-py/test_batch')
