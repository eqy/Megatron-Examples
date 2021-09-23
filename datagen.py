import torch
import os
mode = None
MANUAL_SEED = 42
inds = None
masks = None
data_idx = 0
VOCAB_SIZE = 128
SEQUENCE_LEN = 128
MASK_PROB = 0.1
BATCH_SIZE = 1024
if 'TENSOR_PARALLEL' in os.environ:
    BATCH_SIZE = 128
EASY_MODE = False
EASY_MODE_SIZ = 32

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

fancy_data = download_fancy_data()
print(fancy_data.size(0))
effective_length = fancy_data.size(0) // SEQUENCE_LEN
effective_length = fancy_data.size(0) - SEQUENCE_LEN


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
      
     curr = fancy_data[offset:offset+SEQUENCE_LEN].clone().detach()
     temps.append(curr)
  temp = torch.stack(temps, dim=0).cuda()
  mask = masks[data_idx//BATCH_SIZE]
  mask_not = torch.logical_not(mask)
  data = mask * temp + mask_not*124
  label = temp
  return data, label, mask_not

easy_data = None

