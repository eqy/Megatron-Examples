
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
from megatron import p2p_communication

from megatron.mpu.initialize import get_tensor_model_parallel_group, get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from megatron.mpu.utils import VocabUtility

VOCAB_SIZE = 128
SEQUENCE_LEN = 128
MASK_PROB = 0.1
BATCH_SIZE = 1024
EASY_MODE = False
EASY_MODE_SIZ = 32
STEPS = 10000000
PRINT_INTERVAL = 101
MANUAL_SEED = 42
DEBUG_PRINT = False
NO_RUNAHEAD = False

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
inds = None
masks = None
data_idx = 0

def printtensor(tensor):
  for i in range(tensor.size(0)):
      tmp = tensor[i]
      chars = [chr(tmp[i]) for i in range(len(tmp))]
      text = ''.join(chars)
      print(text)
      print('='*SEQUENCE_LEN)

def post_processing(lm_output, labels):
    output = lm_output
    orig_output = output.clone().detach()
    loss = mpu.vocab_parallel_cross_entropy(output, labels)
    return loss, orig_output

# build a batch given sequence_len and batch size
def generate_fancy_data_labels(sequence_len, batch_size):
  global data_idx
  global inds
  global masks
  global MANUAL_SEED
  temps = list()
  for i in range(batch_size):
     if inds is None or data_idx >= len(inds):
       # hack as use of RNG will fall out of sync due to pipelines being different
       torch.manual_seed(MANUAL_SEED)
       inds = torch.randperm(effective_length, device='cuda')
       masks = (torch.rand(len(inds)//BATCH_SIZE + 1, BATCH_SIZE, SEQUENCE_LEN, device='cuda') >= MASK_PROB).long()
       MANUAL_SEED += 1

       print("new epoch", len(inds))
       data_idx = 0
       print("my start", inds[0:5])
       print("masks_checksum:", torch.sum(masks))
     if EASY_MODE:
       data_idx_ = data_idx % EASY_MODE_SIZ
     else:
       data_idx_ = data_idx
     offset = inds[data_idx_] #* SEQUENCE_LEN
     data_idx += 1
      
     curr = fancy_data[offset:offset+SEQUENCE_LEN].detach().clone()
     temps.append(curr)
  temp = torch.stack(temps, dim=0).cuda()
  mask = masks[data_idx//BATCH_SIZE]
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
model = BertModel(num_tokentypes=0, add_binary_head=False,
                  pre_process=mpu.is_pipeline_first_stage(),
                  post_process=mpu.is_pipeline_last_stage())
if not isinstance(model, list):
    bert = [model]
else:
    bert = model

bert[0] = bert[0].cuda()
fp16 = bert[0].language_model.encoder._get_layer(0).self_attention.fp16
bf16 = bert[0].language_model.encoder._get_layer(0).self_attention.bf16
# even though args are saved in the model, we still need to do this...
if fp16 or bf16:
    bert[0] = Float16Module(bert[0], get_args())

args = get_args()


if args.DDP_impl == 'torch':
    # not supported
    raise Exception
    i = torch.cuda.current_device()
    bert[0] = torchDDP(bert[0], device_ids=[i], output_device=i, process_group=mpu.get_data_parallel_group())
else:
    bert[0] = LocalDDP(bert[0], args.accumulate_allreduce_grads_in_fp32, True)

unwrapped_model = unwrap_model(bert,
			       (torchDDP, LocalDDP, Float16Module))

# Turn on training mode which enables dropout.
for model_module in bert:
    model_module = model_module.train()

for model_module in unwrapped_model:
    model_module = model_module.train()


optimizer = get_megatron_optimizer(unwrapped_model)
# check that this is an FP16 optimizer when args.fp16 is set...
print(type(optimizer))

max_lr=3e-4
min_lr=3e-12
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
    for partition in bert:
        partition.zero_grad_buffer()
    optimizer.zero_grad()

    input_tensors = []
    output_tensors = []
    losses_reduced = []

    num_microbatches = BATCH_SIZE//args.micro_batch_size
    # protocol debugging trackers
    send_forward_count = 0
    recv_forward_count = 0
    send_backward_count = 0
    recv_backward_count= 0

    # rng can be dangerous
    if args.fp16:
      batch_data, batch_label, batch_loss_mask = generate_toy_data_labels(SEQUENCE_LEN, BATCH_SIZE)
    else:
      # print("GENERATING", i)
      batch_data, batch_label, batch_loss_mask = generate_fancy_data_labels(SEQUENCE_LEN, BATCH_SIZE)

    batch_data = batch_data.reshape(batch_data.shape[0]//args.micro_batch_size,
                                    args.micro_batch_size,
                                    batch_data.shape[1])
    batch_label = batch_label.reshape(batch_label.shape[0]//args.micro_batch_size,
                                      args.micro_batch_size,
                                      batch_label.shape[1])
    batch_loss_mask = batch_loss_mask.reshape(batch_loss_mask.shape[0]//args.micro_batch_size,
                                              args.micro_batch_size,
                                              batch_loss_mask.shape[1])

    num_runahead_microbatches = mpu.get_pipeline_model_parallel_world_size() - mpu.get_pipeline_model_parallel_rank() - 1
    num_runahead_microbatches = min(num_runahead_microbatches, num_microbatches)
    if NO_RUNAHEAD:
        num_runahead_microbatches = 0

    if i % PRINT_INTERVAL == 0 and DEBUG_PRINT:
      print(i, "num runahead:", num_runahead_microbatches, "num microbatches:", num_microbatches)
      print(i, "data checksum:", torch.sum(batch_data), "label checksum:", torch.sum(batch_label), "mask_checksum:", torch.sum(batch_loss_mask))

    data_ub_idx = 0

    # runahead to reduce pipeline stalls
    for ub_runahead_warmup in range(num_runahead_microbatches):
        input_tensor = p2p_communication.recv_forward()
        recv_forward_count += 1
        data = batch_data[data_ub_idx]
        label = batch_label[data_ub_idx]
        loss_mask = batch_loss_mask[data_ub_idx]
        data_ub_idx += 1

        # no padding in either toy or "fancy" data
        padding_mask = torch.ones_like(data, device='cuda')

        # start of "forward_step function"
        unwrapped_model = unwrap_model(bert[0], (torchDDP, LocalDDP, Float16Module))
        unwrapped_model.set_input_tensor(input_tensor)
        output_tensor = bert[0](data, padding_mask, tokentype_ids=None, lm_labels=None)
        if DEBUG_PRINT and mpu.is_pipeline_first_stage() and i % PRINT_INTERVAL == 0:
            print("first stage did inference on data:")
            printtensor(data[:1,:])
        assert not mpu.is_pipeline_last_stage()
        # end of "forward_step function"
        p2p_communication.send_forward(output_tensor)
        send_forward_count += 1
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    if num_runahead_microbatches < num_microbatches:
        input_tensor = p2p_communication.recv_forward()
        recv_forward_count += 1
     
    for ub in range(num_runahead_microbatches, num_microbatches):
        #if DEBUG_PRINT and i % PRINT_INTERVAL == 0:
        #    print("rank", mpu.get_pipeline_model_parallel_rank(), ub)
        last_iteration = (ub == (num_microbatches - 1))

        data = batch_data[data_ub_idx]
        label = batch_label[data_ub_idx]
        loss_mask = batch_loss_mask[data_ub_idx]
        data_ub_idx += 1

        # no padding in either toy or "fancy" data
        padding_mask = torch.ones_like(data, device='cuda')

        # start of "forward_step function"
        unwrapped_model = unwrap_model(bert[0], (torchDDP, LocalDDP, Float16Module))
        unwrapped_model.set_input_tensor(input_tensor)
        output_tensor = bert[0](data, padding_mask, tokentype_ids=None, lm_labels=None)
        #if DEBUG_PRINT and mpu.is_pipeline_first_stage() and i % PRINT_INTERVAL == 0:
        #    print("first stage did inference on data:")
        #    printtensor(data[:1,:])
        if mpu.is_pipeline_last_stage():
            output_tensor, _ = output_tensor
            output_tensor, orig_output = post_processing(output_tensor, label)
            lm_loss_ = output_tensor
            lm_loss_ = lm_loss_.float()
            loss_mask = loss_mask.float()
            lm_loss = torch.sum(
                lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
            prescale_loss = lm_loss / num_microbatches
            output_tensor = prescale_loss
            #if DEBUG_PRINT and i % PRINT_INTERVAL == 0:
            #    print("last stage did loss on label:", f"({data_ub_idx})")
            #    printtensor(label[:1,:])
        # end of "forward_step function"
        output_tensor_grad = p2p_communication.send_forward_recv_backward(output_tensor)
        send_forward_count += 1
        recv_backward_count += 1

        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

        input_tensor = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)
        
        # start of "backward_step function"
        if input_tensor is not None:
            input_tensor.retain_grad()
        if output_tensor_grad is None:
            output_tensor = optimizer.scale_loss(output_tensor)
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)
        input_tensor_grad = None
        if input_tensor is not None:
            input_tensor_grad = input_tensor.grad
        # end of "backward_step function"
        #if DEBUG_PRINT and i % PRINT_INTERVAL == 0:
        #    if input_tensor_grad is not None:
        #        print("steady state grad sum:", torch.sum(input_tensor_grad))
        #    else:
        #        print("none in steady-state")

        if last_iteration:
            input_tensor = None
            p2p_communication.send_backward(input_tensor_grad)
            send_backward_count += 1
        else:
            input_tensor = p2p_communication.send_backward_recv_forward(input_tensor_grad)
            send_backward_count += 1
            recv_forward_count += 1

    for ub_runahead_cooldown in range(num_runahead_microbatches):
        output_tensor_grad = p2p_communication.recv_backward()
        recv_backward_count += 1
        input_tensor = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)

        # start of "backward_step function"
        if input_tensor is not None:
            input_tensor.retain_grad()
        if output_tensor_grad is None:
            output_tensor = optimizer.scale_loss(output_tensor)
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)
        input_tensor_grad = None
        if input_tensor is not None:
            input_tensor_grad = input_tensor.grad
        # end of "backward_step function"
        #if DEBUG_PRINT and i % PRINT_INTERVAL == 0:
        #    if input_tensor_grad is not None:
        #        print("tail grad sum:", torch.sum(input_tensor_grad))
        #    else:
        #        print("none in tail")
        p2p_communication.send_backward(input_tensor_grad)
        send_backward_count += 1

    assert len(input_tensors) == 0
    assert len(output_tensors) == 0
    # assert data_ub_idx == num_microbatches

    if args.DDP_impl == 'local':
        bert[0].allreduce_gradients()

    if (mpu.is_pipeline_first_stage(ignore_virtual=True) or
        mpu.is_pipeline_last_stage(ignore_virtual=False)) and \
            mpu.get_pipeline_model_parallel_world_size() > 1:
        assert bert[0] == bert[-1]
        unwrapped_model = unwrap_model(bert[0], (torchDDP, LocalDDP, Float16Module))
        if unwrapped_model.share_word_embeddings:
            word_embeddings_weight = unwrapped_model.word_embeddings_weight()
            grad = word_embeddings_weight.main_grad 
            torch.distributed.all_reduce(grad, group=mpu.get_embedding_group())

    # fp16_lm_cross_entropy False by default, so we add a cast to float here
    # used if we turn off postprocessing in the bert model
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    if update_successful:
        lr_scheduler.step(BATCH_SIZE)
    else:
        # indicates NaNs in the gradients/loss
        print("update failed")

    if i % PRINT_INTERVAL == 0:
        if mpu.is_pipeline_last_stage():
            printtensor(data[:2,])
            preds = torch.argmax(orig_output, dim=2) * loss_mask
            cleaned = data * torch.logical_not(loss_mask)
            printtensor((cleaned + preds)[:2,:].long())
            print("LABEL")
            printtensor(label[:2, :])
            print((i+1)*BATCH_SIZE, "UNSCALED LOSS:", lm_loss)
        if mpu.is_pipeline_first_stage():
            print((i+1)*BATCH_SIZE/(time.time() - start_time), "samples/sec")
        if DEBUG_PRINT and i % PRINT_INTERVAL == 0:
            print(mpu.get_pipeline_model_parallel_rank(), {'send_forward_count':send_forward_count,
                                                           'recv_forward_count':recv_forward_count,
                                                           'send_backward_count':send_backward_count,
                                                           'recv_backward_count':recv_backward_count})
