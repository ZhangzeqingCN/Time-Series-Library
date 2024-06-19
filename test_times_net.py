from models import TimesNet
from utils.args import Args

model = TimesNet.Model(Args())
model.train()
