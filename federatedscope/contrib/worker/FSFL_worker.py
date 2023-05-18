from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client
from federatedscope.core.message import Message
import logging
from torch.utils.data import Dataset, DataLoader
from federatedscope.contrib.common_utils import get_public_dataset


logger = logging.getLogger(__name__)


# Build your worker here.
class FSFL_Client(Client):
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):
        super(FSFL_Client, self).__init__(ID, server_id, state, config, data, model, device,
                                          strategy, is_unseen_client, *args, **kwargs)

    def callback_funcs_for_local_pretraining(self, message: Message):
        round = message.state
        content = message.content
        cfg = self.cfg
        dataset = cfg.MHFL.public_dataset
        task = cfg.MHFL.task
        train_batch_size = cfg.MHFL.public_batch_size
        test_batch_size = cfg.MHFL.public_batch_size

        if task == 'CV':
            train_data, test_data = get_public_dataset(dataset)
            train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)


        #train on public dataset
        model =self.model
        model.to(self.device)
        model.train()
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=0.5)
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=1e-4)
        trainloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        criterion = torch.nn.NLLLoss().to(device)
        train_epoch_losses = []


    def callback_funcs_for_model_para(self, message: Message):
        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content

        self.trainer.update(content)
        self.state = round

        sample_size, model_para, results, agg_protos = self.trainer.train()

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, agg_protos)))


def call_my_worker(method):
    if method == 'fedproto':
        worker_builder = {'client': FedprotoClient, 'server': FedprotoServer}
        return worker_builder


register_worker('fedproto', call_my_worker)
