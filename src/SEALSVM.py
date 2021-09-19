from SVM import SVM

class SEALSVM(SVM):
    def __init__(self, name: str, learning_rate=0.001, lambda_param=0.01, n_iters=1000) -> None:
        super().__init__(name, learning_rate=learning_rate, lambda_param=lambda_param, n_iters=n_iters)