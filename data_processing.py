from torch.utils import data
from fusenet_solver import Solver
from utils.data_utils import get_data
#from utils.loss_utils import cross_entropy_2d
from utils.uniform_loss_utils import cross_entropy_2d
from options.train_options import TrainOptions
from utils.utils import print_time_info

if __name__ == '__main__':
	opt = TrainOptions().parse()
	train_data, test_data = get_data(opt, use_train=True, use_test=True)
	print(test_data[0])