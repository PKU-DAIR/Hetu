NCCL_DEBUG=INFO mpirun --allow-run-as-root -np 8 -x PYTHONPATH=/home/Hetu/python python test_moe_ktop1.py --k=2 --num_local_experts=2 --batch_size=64
