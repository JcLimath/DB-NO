import os, sys, time
import copy
import numpy as np
import argparse
import random
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import wandb
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_utils import get_data_loader
from utils.optimizer_utils import set_scheduler, set_optimizer
from utils.loss_utils import LossMSE
from utils.misc_utils import compute_grad_norm, vis_fields_many, l2_err, show
from utils.domains import DomainXY
from collections import OrderedDict

# models
import models.fno

def print_mem():
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

def set_seed(params, world_size):
    seed = params.seed
    if seed is None:
        seed = np.random.randint(10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if world_size > 0:
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

class Inferencer():
    """ trainer class """
    def __init__(self, params, args):
        self.sweep = args.sweep
        self.root_dir = args.root_dir
        self.config = args.config 
        self.run_num = args.run_num
        self.world_size = 1
        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])

        self.local_rank = 0
        self.world_rank = 0
        if self.world_size > 1:
            dist.init_process_group(backend='nccl',
                                    init_method='env://')
            self.world_rank = dist.get_rank()
            self.local_rank = int(os.environ["LOCAL_RANK"])

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            torch.backends.cudnn.benchmark = True
        
        self.log_to_screen = params.log_to_screen and self.world_rank==0
        self.log_to_wandb = False # turn off for inference; params.log_to_wandb and self.world_rank==0
        params['name'] = args.config + '_' + args.run_num
        params['group'] = 'op_' + args.config
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device('cpu')
        self.params = params
        self.params.device = self.device
        self._build()

    def _build(self):
        # init wandb
        if self.sweep != 'none':
            exp_dir = os.path.join(*[self.root_dir, 'sweeps', self.sweep, self.config, 'inference'])
        else:
            exp_dir = os.path.join(*[self.root_dir, 'expts', self.config, self.run_num])

        if self.world_rank==0:
            if not os.path.isdir(exp_dir):
                os.makedirs(exp_dir)
                os.makedirs(os.path.join(exp_dir, 'wandb/'))

        self.params['experiment_dir'] = os.path.abspath(exp_dir)
        if self.log_to_wandb:
            wandb.init(dir=os.path.join(exp_dir, "wandb"),
                        config=self.params.params, name=self.params.name, group=self.params.group, project=self.params.project, 
                        entity=self.params.entity)

        set_seed(self.params, self.world_size)

        self.params['global_batch_size'] = self.params.batch_size
        self.params['local_batch_size'] = int(self.params.batch_size//self.world_size)
        self.params['global_valid_batch_size'] = self.params.valid_batch_size
        self.params['local_valid_batch_size'] = int(self.params.valid_batch_size//self.world_size)

        if self.world_rank==0:
            self.params.log()

    def launch(self):
        self.test_data_loader, self.test_dataset, _ = get_data_loader(self.params, self.params.test_path, dist.is_initialized(), train=False, pack=self.params.pack_data)

        # domain grid
        self.domain = DomainXY(self.params)
        
        if self.params.model == 'fno':
            self.model = models.fno.fno(self.params).to(self.device)
        else:
            assert(False), "Error, model arch invalid."

        if dist.is_initialized():
            self.model = DistributedDataParallel(self.model,
                                                device_ids=[self.local_rank],
                                                output_device=[self.local_rank])

        if self.params.loss_func == "mse":
            self.loss_func = LossMSE(self.params, self.model)
        else:
            assert(False), "Error,  loss func invalid."

        if hasattr(self.params, 'weights'):
            self.params['checkpoint_path'] = self.params['weights']
            logging.info("Loading checkpoint %s"%self.params.checkpoint_path)
            self.restore_checkpoint(self.params.checkpoint_path)

        if self.log_to_screen:
            print("model wt norm = {}".format(self.get_model_wt_norm(self.model)))

        self.logs = {}
        test_time, fields = self.test()
        logging.info("testing time = {}".format(test_time))
        if self.log_to_wandb:
            # log visualizations every epoch
            fig = vis_fields_many(fields, self.params, self.domain)
            self.logs['vis'] = wandb.Image(fig)
            plt.close(fig)
            wandb.log(self.logs, step=1)
            wandb.finish()

    def get_model_wt_norm(self, model):
        n = 0
        for p in model.parameters():
            p_norm = p.data.detach().norm(2)
            n += p_norm.item()**2
        n = n**0.5
        return n

    def test(self):
        self.model.eval() 
        #self.model.train() # need gradients
        test_start = time.time()

        logs_buff = torch.zeros((2), dtype=torch.float32, device=self.device)
        self.logs['test_err'] = logs_buff[0].view(-1)
        self.logs['test_loss'] = logs_buff[1].view(-1)

        num_examples = 3
        idx = [np.random.randint(0, len(self.test_data_loader)) for _ in range(num_examples)] # index of batch
        img_idx = [np.random.randint(0, self.params.local_valid_batch_size) for _ in range(num_examples)] # index within batch
        fields = []
        ii = 0

        bs = self.params.local_valid_batch_size
        
        import scipy.ndimage

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.test_data_loader):
                if not self.params.pack_data:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                u = self.model(inputs)
                
                '''
                误差图              
                '''
                number_plot = 0
                #task = 'po-FNO-TS-SD_abs.png'
                #task = 'po-FNO-TS_abs.png'
                #task = 'po-FNO_abs.png'
                task = 'po-target.png'
                
                
                abs_error = torch.abs(u[number_plot]-targets[number_plot])
                
                abs_error_np = targets[number_plot].squeeze().cpu().numpy()
                x = np.linspace(0, 1, abs_error_np.shape[1])
                y = np.linspace(0, 1, abs_error_np.shape[0])
                
                plt.figure(figsize=(8, 6))
                plt.imshow(abs_error_np, extent=[0, 1, 0, 1], cmap='rainbow', interpolation='bilinear', vmin=0, vmax=None)
                #cbar = plt.colorbar(ticks=np.linspace(0, 0.006, num=5))
                cbar = plt.colorbar()
                cbar.ax.set_ylabel('',fontsize=23)
                cbar.ax.tick_params(labelsize=23)
            
                plt.xlabel('X', fontsize=25)
                plt.ylabel('Y', fontsize=25)
                plt.xticks(fontsize=23)
                plt.yticks(fontsize=23)
                #plt.title('FNO')
                plt.savefig(task, bbox_inches='tight')
                print(l2_err(u[number_plot].detach(), targets[number_plot].detach()))
                stop()
                
                
                
                
                
                
                
                loss_data = self.loss_func.data(inputs, u, targets)
                loss_pde = self.loss_func.pde(inputs, u, targets)
                loss_bc = self.loss_func.bc(inputs, u, targets)
                loss = loss_data + loss_bc + loss_pde

                self.logs['test_err'] += l2_err(u.detach(), targets.detach()) # computes rel l2 err of each image and averages across batches
                self.logs['test_loss'] += loss.detach()
                if i in idx: 
                    source = inputs[img_idx[ii],0].detach().cpu().numpy() 
                    soln = targets[img_idx[ii],0].detach().cpu().numpy()
                    pred = u[img_idx[ii],0].detach().cpu().numpy()
                    fields.extend([source, soln, pred])
                    ii += 1

        self.logs['test_err'] /= len(self.test_data_loader)
        self.logs['test_loss'] /= len(self.test_data_loader)

        if dist.is_initialized():
            for key in ['test_loss', 'test_err']:
                dist.all_reduce(self.logs[key].detach())
                self.logs[key] = float(self.logs[key]/dist.get_world_size())
        else:
            for key in ['test_loss', 'test_err']:
                self.logs[key] = float(self.logs[key])

        if self.log_to_screen:
            print(self.logs)

        #self.save_logs(tag="_ckpt")
        self.save_logs(tag="_best")

        test_time = time.time() - test_start

        return test_time, fields

    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.local_rank)) 
        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                name = key[7:]
                new_state_dict[name] = val 
            self.model.load_state_dict(new_state_dict)


    def save_logs(self, tag=""):
        with open(os.path.join(self.params.experiment_dir, "logs"+tag+".txt"), "w") as f:
            for k, v in self.logs.items():
                f.write("{},{}\n".format(k,v))
