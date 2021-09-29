# help visualize Megatron's communication pattern by printing rank groups
world_size = 8
pipeline_model_parallel_size = 2
tensor_model_parallel_size = 2
data_parallel_size = world_size // (pipeline_model_parallel_size * tensor_model_parallel_size)

data_parallel_groups = list()
tensor_model_parallel_groups = list()
pipeline_model_parallel_groups = list()

num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size
print("DATA PARALLEL GROUPS (need to communicate with everyone in group)")
# each pipeline stage needs to sync gradients with corresponding data-parallel group
for i in range(pipeline_model_parallel_size):
    # but not if they are in the same pipeline-parallel group
    start_rank = i * num_pipeline_model_parallel_groups
    end_rank = (i+1) * num_pipeline_model_parallel_groups
    # and not if they are in the same model-parallel group
    for j in range(tensor_model_parallel_size):
        ranks = range(start_rank + j, end_rank, tensor_model_parallel_size)
        group = [x for x in ranks]
        data_parallel_groups.append(group)
        print(group)

print("TENSOR MODEL PARALLEL GROUPS (need to communicate with everyone in group)")
# each model parallel shard needs to allreduce outputs with corresponding model-parallel group
for i in range(num_tensor_model_parallel_groups):
    # we want immediate neighbors so don't care about other dimensions
    ranks = range(i * tensor_model_parallel_size, (i+1)*tensor_model_parallel_size)
    group = [x for x in ranks]
    tensor_model_parallel_groups.append(group)
    print(group)

print("PIPELINE MODEL PARALLEL GROUPS (need to communicate with neighbors in group)")
# each pipeline stage needs to wait on inputs/outputs with corresponding pipeline-parallel group
for i in range(num_pipeline_model_parallel_groups):
    # we want other ranks in the same stage corresponding, so step by... the number of ranks at the same stage
    ranks = range(i, world_size, num_pipeline_model_parallel_groups)
    group = [x for x in ranks]
  pipeline_model_parallel_groups.append(group)

for i in range(data_parallel_size):
    print("DATA PARALLEL", i)
    for j in range(pipeline_model_parallel_size): # really num of pipeline stages
        print(f'stage {j} {tensor_model_parallel_groups[j*data_parallel_size + i]}')  
