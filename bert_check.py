
import torch

import time


from megatron import get_args
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

VOCAB_SIZE = 128
SEQUENCE_LEN = 128
MASK_PROB = 0.1
BATCH_SIZE = 128
EASY_MODE = False 
STEPS = 5000000

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
       data_idx = data_idx % 64
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

def generate_toy_data_labels(sequence_len, batch_size):
  start = 64
  end = 96
  temp = torch.randint(start, end, (batch_size, sequence_len))
  mask = (torch.rand(batch_size, sequence_len) >= MASK_PROB).long()
  mask_not = torch.logical_not(mask)
  data = mask * temp + mask_not*124
  label = mask * temp + torch.max(temp)*mask_not
  return data, label, mask_not

initialize_megatron()
global_vars._GLOBAL_ARGS.padded_vocab_size = VOCAB_SIZE
bert = [BertModel(num_tokentypes=0, add_binary_head=False, post_process=True)]


bert[0] = bert[0].cuda()
fp16 = bert[0].language_model.encoder._get_layer(0).self_attention.fp16
bf16 = bert[0].language_model.encoder._get_layer(0).self_attention.bf16
# even though args are saved in the model, we still need to do this...
if fp16 or bf16:
    bert[0] = Float16Module(bert[0], get_args())

args = get_args()


if args.DDP_impl == 'torch':
    i = torch.cuda.current_device()
    bert[0] = torchDDP(bert[0], device_ids=[i], output_device=i, process_group=mpu.get_data_parallel_group())
else:
    bert[0] = LocalDDP(bert[0], args.accumulate_allreduce_grads_in_fp32, True)

unwrapped_model = unwrap_model(bert,
			       (torchDDP, LocalDDP, Float16Module))

# Turn on training mode which enables dropout.
bert[0] = bert[0].train()
unwrapped_model[0] = unwrapped_model[0].train()

optimizer = get_megatron_optimizer(unwrapped_model)
# check that this is an FP16 optimizer when args.fp16 is set...
print(type(optimizer))

max_lr=3e-4
min_lr=3e-10
warmup_steps=1000
decay_steps=5000000

# abandoned fp16 tuning for "fancy" data training
if fp16:
    nothing = None
    #warmup_steps *= 100
    #warmup_steps *= 10
    #decay_steps *= 3
    #max_lr = 3e1
    #min_lr = 3e-2
    #BATCH_SIZE *= 2
    #STEPS *= 2
    #min_lr /= 100
    # max_lr *= 10

lr_scheduler = AnnealingLR(
    optimizer,
    max_lr=max_lr,
    min_lr=min_lr,
    warmup_steps=warmup_steps,
    decay_steps=decay_steps,
    decay_style='linear',
    use_checkpoint_lr_scheduler=False,
    override_lr_scheduler=False)

loss_sum = 0.0

print_rank_0(f"fp16: {fp16}, bf16: {bf16}")

start_time = time.time()
# manually set postprocess to false to use loss criterion explicitly
#bert[0].post_process = False
for i in range(STEPS//BATCH_SIZE):
    # rng can be dangerous
    if args.fp16:
      data, label, loss_mask = generate_toy_data_labels(SEQUENCE_LEN, BATCH_SIZE)
    else:
      data, label, loss_mask = generate_fancy_data_labels(SEQUENCE_LEN, BATCH_SIZE)

    label = label.cuda()
    data = data.cuda()
    loss_mask= loss_mask.cuda()
    # no padding in either toy or "fancy" data
    padding_mask = torch.ones_like(data, device='cuda')

    # alternative version where we don't use megatron's "postprocessing" in models, which means that we do the loss calculation on our own 
    output_tensor, _ = bert[0](data, padding_mask, tokentype_ids=None, lm_labels=None)
    # lm_loss_ , _ = bert[0](data, padding_mask, tokentype_ids=None, lm_labels=label)

    # copy-pasted from Megatron train function
    if args.DDP_impl == 'local':
        for partition in bert:
            partition.zero_grad_buffer()
    optimizer.zero_grad()

    if args.DDP_impl == 'local':
        bert[0].allreduce_gradients()

    # fp16_lm_cross_entropy False by default, so we add a cast to float here
    # used if we turn off postprocessing in the bert model
    lm_loss_ = mpu.vocab_parallel_cross_entropy(output_tensor.float(), label)

    lm_loss_ = lm_loss_.float()
    loss_mask = loss_mask.float()
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
    lm_loss.backward()
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    if update_successful:
        lr_scheduler.step(BATCH_SIZE)
    else:
        # indicates NaNs in the gradients/loss
        print("update failed")

    curr_loss = lm_loss.detach().item()
    loss_sum += curr_loss
    if i % 100 == 0:
        s = 0.0
        # to "inspect" the model outputs, we run inference with lm_labels=None as this is the simplest way to get the model output
        # when it is run with lm_labels, only the loss is returned, not the logits
        output_tensor, _ = bert[0](data, padding_mask, tokentype_ids=None, lm_labels=None)
        # gather the model output across the entire vocab (across other GPUs)
        all_vocab = torch.zeros(BATCH_SIZE, SEQUENCE_LEN, VOCAB_SIZE, device='cuda')
        vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_per_partition_vocab_size(output_tensor.size(-1), get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()) 
        all_vocab[:,:,vocab_start_index:vocab_end_index] = output_tensor
        torch.distributed.all_reduce(all_vocab, group=get_tensor_model_parallel_group())
    
        for key in bert[0].state_dict():
            s += torch.sum(bert[0].state_dict()[key].float())
        print_rank_0([i, i*BATCH_SIZE, "LOSS:", curr_loss, "AVG LOSS:", loss_sum/(i+1)])
        print_rank_0(["vocab range", vocab_start_index, vocab_end_index])
        print_rank_0(all_vocab.shape)
        print_rank_0(["param sum", s, "loss scale", optimizer.get_loss_scale()])

        # show some examples in the output
        printtensor(data[:2,])
        preds = torch.argmax(all_vocab, dim=2) * loss_mask
        cleaned = data * torch.logical_not(loss_mask)
        printtensor((cleaned + preds)[:2,:].long())
        # assumes printing has effectively caused a CUDA synchronize
        curr_elapsed = time.time() - start_time
        print_rank_0(f"{(i+1)*BATCH_SIZE/curr_elapsed} samples/sec")
