NCCL_DEBUG=INFO mpirun --allow-run-as-root -np 2 -x  python ../test_mnist.py --top=1 --num_local_experts=2 --batch_size=16
