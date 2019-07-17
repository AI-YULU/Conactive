class Config(object):
  def __init__(self):
    self.learning_rate = 0.0001
    self.batch_size = 512
    self.win_size = 400
    self.n_traces = 1
    self.display_step = 50
    self.n_threads = 2
    self.n_epochs = None
    self.regularization = 1e-3

    # Number of epochs, None is infinite
    n_epochs = None
