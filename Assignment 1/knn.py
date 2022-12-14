def knn(x_train, y_train, x_test, n_classes, device):
    """
    x_train: 60000 x 784 matrix: each row is a flattened image of an MNIST digit
    y_train: 60000 vector: label for x_train
    x_test: 1000 x 784 testing images
    n_classes: no. of classes in the classification task
    device: pytorch device on which to run the code
    return: predicted y_test which is a 1000-sized vector
    """
    """
    x_train: 60000 x 784 matrix: each row is a flattened image of an MNIST digit
    y_train: 60000 vector: label for x_train
    x_test: 5000 x 784 testing images
    return: predicted y_test which is a 5000 vector
    """
    # convert data to tensors
    xt_test = torch.tensor(x_test, dtype=torch.float, device=device)
    xt_train = torch.tensor(x_train, dtype=torch.float, device=device)
    yt_train = torch.tensor(y_train, dtype=torch.float, device=device)

    # return: predicted y_test which is a 5000 vector
    y_test = torch.zeros(len(x_test), device=device)

    # find distance of each test image from all training images
    for i in range(len(xt_test)):
      dist = xt_train - xt_test[i]
      
      # indices of k training images with the smallest distances, largest=False takes the smallest
      topk, idx = torch.topk(torch.linalg.norm(dist, dim=1), k=5, largest = False)

      # classes
      cls = torch.gather(yt_train, 0, idx)

      # most frequent cls
      # m_cls =  torch.argmax(cls)

      # as one hot vector of size 10
      vect = torch.nn.functional.one_hot(cls.to(torch.int64), n_classes)

      # col wise sum of this of this vect array
      s = torch.sum(vect, 0)

      # col with max sum
      y_test[i] = torch.argmax(s).float()


    return y_test.cpu().numpy()
