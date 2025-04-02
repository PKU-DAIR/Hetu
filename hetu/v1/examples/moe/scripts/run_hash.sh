NCCL_DEBUG=INFO mpirun --allow-run-as-root -np 8 -x PYTHONPATH=/home/Hetu/python python test_moe_hash.py --num_local_experts=2 --batch_size=4
