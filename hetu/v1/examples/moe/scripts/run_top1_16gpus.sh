# change ht.alltoall_op to ht.halltoall_op in Hetu/python/hetu/layers/moe_layer.py if you want to use 
# hierarchical AllToAll.

NCCL_DEBUG=DEBUG mpirun --allow-run-as-root -np 16 -mca btl_tcp_if_include enp97s0f0 -x NCCL_SOCKET_IFNAME=enp97s0f0  -x PYTHONPATH=/home/Hetu/python -H node1:8,node2:8 /root/anaconda3/envs/moe/bin/python /home/Hetu/tests/test_moe_top.py --top=1 --num_local_experts=1 --batch_size=1
