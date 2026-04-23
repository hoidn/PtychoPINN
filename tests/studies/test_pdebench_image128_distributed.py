import torch


def test_initialize_runtime_configures_cuda_ddp_from_torchrun_env(monkeypatch):
    from scripts.studies.pdebench_image128 import distributed

    init_calls = []
    set_device_calls = []

    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(distributed.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(distributed.torch.cuda, "set_device", lambda index: set_device_calls.append(index))
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(distributed.dist, "init_process_group", lambda backend: init_calls.append(backend))

    runtime = distributed.initialize_runtime("cuda")

    assert runtime.distributed_enabled is True
    assert runtime.rank == 2
    assert runtime.local_rank == 1
    assert runtime.world_size == 4
    assert runtime.backend == "nccl"
    assert runtime.device.type == "cuda"
    assert runtime.device.index == 1
    assert runtime.launched_via_torchrun is True
    assert init_calls == ["nccl"]
    assert set_device_calls == [1]


def test_distributed_runtime_build_training_loader_uses_distributed_sampler():
    from scripts.studies.pdebench_image128.distributed import DistributedRuntime

    runtime = DistributedRuntime(
        requested_device="cpu",
        device=torch.device("cpu"),
        rank=1,
        local_rank=1,
        world_size=2,
        backend="gloo",
        distributed_enabled=True,
        launched_via_torchrun=True,
    )

    dataset = [{"input": torch.zeros(1, 8, 8), "target": torch.zeros(1, 8, 8), "sample_index": 0}] * 4
    loader, sampler = runtime.build_training_loader(
        dataset,
        batch_size=2,
        num_workers=0,
        collate_fn=lambda batch: {"items": batch},
        shuffle=False,
    )

    assert sampler is not None
    assert loader.sampler is sampler
    assert loader.batch_size == 2
