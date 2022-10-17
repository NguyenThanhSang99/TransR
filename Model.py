import torch
import torch.nn as nn

class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
        self.zero_const = nn.Parameter(torch.Tensor([0]))
        self.zero_const.requires_grad = False
    


class Model(BaseModule):

	def __init__(self, ent_num, rel_num):
		super(Model, self).__init__()
		self.ent_num = ent_num
		self.rel_num = rel_num

	def forward(self):
		raise NotImplementedError
	
	def predict(self):
		raise NotImplementedError