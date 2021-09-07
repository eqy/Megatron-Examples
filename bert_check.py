import time

import torch

from megatron.model import Float16Module
from megatron.optimizer import get_megatron_optimizer 
from megatron.utils import unwrap_model

from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model import DistributedDataParallel as LocalDDP

from megatron.learning_rates import AnnealingLR

from megatron.model.bert_model import BertModel
from megatron.utils import average_losses_across_data_parallel_group
from megatron import initialize_megatron
from megatron import global_vars

from megatron import mpu
from megatron import print_rank_0

from megatron.mpu.initialize import get_tensor_model_parallel_group, get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from megatron.mpu.utils import VocabUtility

# "fancy" data

VOCAB_SIZE = 128
SEQUENCE_LEN = 128
MASK_PROB = 0.1
BATCH_SIZE = 256
EASY_MODE = False

torch.manual_seed(42)

# download a public domain book as corpus
def download_fancy_data():
  import requests
  response = requests.get('https://www.gutenberg.org/files/1342/1342-0.txt')
  #response = requests.get('https://www.gutenberg.org/files/84/84-0.txt')
  text = ' '.join(response.text.split())
  encoded = text.encode('ascii', 'replace')
  ints = [int(encoded[i]) for i in range(len(encoded))]
  return torch.tensor(ints)

fancy_data = download_fancy_data()
print(fancy_data.size(0))
effective_length = fancy_data.size(0) // SEQUENCE_LEN
effective_length = fancy_data.size(0) - SEQUENCE_LEN
inds = torch.randperm(effective_length)
data_idx = 0

def printtensor(tensor):
  for i in range(tensor.size(0)):
      tmp = tensor[i]
      chars = [chr(tmp[i]) for i in range(len(tmp))]
      text = ''.join(chars)
      print_rank_0(text)
      print_rank_0('='*SEQUENCE_LEN)

# build a batch given sequence_len and batch size
def generate_fancy_data_labels(sequence_len, batch_size):
  global data_idx
  global inds
  temps = list()
  for i in range(batch_size):
     #offset = torch.randint(fancy_data.size(0) - SEQUENCE_LEN, (1,))[0]
     if data_idx >= len(inds):
       print("new epoch", len(inds))
       data_idx = 0
       inds = torch.randperm(effective_length)
     if EASY_MODE:
       data_idx = data_idx % 5
     offset = inds[data_idx] #* SEQUENCE_LEN
     data_idx += 1
      
     curr = fancy_data[offset:offset+SEQUENCE_LEN].detach().clone()
     temps.append(curr)
  temp = torch.stack(temps, dim=0)
  mask = ( torch.rand(batch_size, sequence_len) >= MASK_PROB ).long()
  mask_not = torch.logical_not(mask)
  data = mask * temp + mask_not*124
  label = temp
  return data, label, mask_not


initialize_megatron()
global_vars._GLOBAL_ARGS.padded_vocab_size = VOCAB_SIZE
bert = BertModel(num_tokentypes=0, add_binary_head=False)
bert = bert.cuda()

unwrapped_model = unwrap_model(bert,
			       (torchDDP, LocalDDP, Float16Module))

# Turn on training mode which enables dropout.
bert = bert.train()
unwrapped_model = unwrapped_model.train()

optimizer = get_megatron_optimizer([unwrapped_model])

lr_scheduler = AnnealingLR(
    optimizer,
    max_lr=3e-4,
    min_lr=3e-10,
    warmup_steps=10000,
    decay_steps=1000000,
    decay_style='linear',
    use_checkpoint_lr_scheduler=False,
    override_lr_scheduler=False)

loss_sum = 0.0

start_time = time.time()
for i in range(5000000//BATCH_SIZE):
    generate_fancy_data_labels(SEQUENCE_LEN, BATCH_SIZE)
    # rng can be dangerous
    data, label, loss_mask = generate_fancy_data_labels(SEQUENCE_LEN, BATCH_SIZE)

    label = label.cuda()
    data = data.cuda()
    loss_mask= loss_mask.cuda()
    padding_mask = torch.ones_like(data, device='cuda')
    # print(data.size(0))
    # if i % 10 == 0:
    #     print("iter", i, "I am rank", get_tensor_model_parallel_rank(), data[0, :5])
    output_tensor, _ = bert(data, padding_mask, tokentype_ids=None, lm_labels=None)
    output_clone = output_tensor.detach().clone()
    all_vocab = torch.zeros(BATCH_SIZE, SEQUENCE_LEN, VOCAB_SIZE, device='cuda')
    vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_per_partition_vocab_size(output_clone.size(-1), get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size())

    all_vocab[:,:,vocab_start_index:vocab_end_index] = output_clone
    torch.distributed.all_reduce(all_vocab, group=get_tensor_model_parallel_group())
    optimizer.zero_grad()

    lm_loss_ = mpu.vocab_parallel_cross_entropy(output_tensor, label)
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
    lm_loss.backward()
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    if update_successful:
        lr_scheduler.step(BATCH_SIZE)
    else:
        print("update failed")

    curr_loss = lm_loss.detach().item()
    loss_sum += curr_loss
    if i % 100 == 0:
        s = 0.0
        for key in unwrapped_model.state_dict():
            s += torch.sum(bert.state_dict()[key])
        print_rank_0([i, i*BATCH_SIZE, "LOSS:", curr_loss, "AVG LOSS:", loss_sum/(i+1)])
        print_rank_0(["vocab range", vocab_start_index, vocab_end_index])
        print_rank_0(all_vocab.shape)

        printtensor(data[:2,])
        preds = torch.argmax(all_vocab, dim=2) * loss_mask
        cleaned = data * torch.logical_not(loss_mask)
        printtensor((cleaned + preds)[:2,:])
        # assumes printing has effectively caused a CUDA synchronize
        curr_elapsed = time.time() - start_time
        print_rank_0(f"{(i+1)*BATCH_SIZE/curr_elapsed} samples/sec")
