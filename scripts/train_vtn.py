#!/usr/bin/env python3
"""Train the VTN direct-VC model on the SAP Lhotse manifests.

Config-driven; all paths are explicit (cluster-friendly). Example:

    .venv/bin/python scripts/train_vtn.py \
        --manifest-dir /projects/aanchan/data/manifests \
        --exp-dir /scratch/aa.mohan/exp/vtn_base \
        --batch-size 8 --lr 1e-3 --max-steps 100000

Quick smoke (tiny model, few utterances, CPU):

    .venv/bin/python scripts/train_vtn.py --exp-dir /tmp/vtn_smoke \
        --max-train-ids 16 --max-steps 50 --val-every 0 --num-workers 0 --d-model 64
"""
import argparse

from sap.models.vc.vtn.model import VTNConfig
from sap.models.vc.vtn.trainer import TrainConfig, train


def main():
    p = argparse.ArgumentParser(description="Train VTN direct VC on SAP manifests")
    p.add_argument("--manifest-dir", default=TrainConfig.manifest_dir)
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--val-batches", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-train-ids", type=int, default=None)
    p.add_argument("--max-val-ids", type=int, default=None)
    # model size
    p.add_argument("--d-model", type=int, default=VTNConfig.d_model)
    p.add_argument("--nhead", type=int, default=VTNConfig.nhead)
    p.add_argument("--enc-layers", type=int, default=VTNConfig.num_encoder_layers)
    p.add_argument("--dec-layers", type=int, default=VTNConfig.num_decoder_layers)
    p.add_argument("--ffn", type=int, default=VTNConfig.dim_feedforward)
    args = p.parse_args()

    model = VTNConfig(d_model=args.d_model, nhead=args.nhead,
                      num_encoder_layers=args.enc_layers, num_decoder_layers=args.dec_layers,
                      dim_feedforward=args.ffn)
    cfg = TrainConfig(
        manifest_dir=args.manifest_dir, train_split=args.train_split, val_split=args.val_split,
        exp_dir=args.exp_dir, model=model, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip, max_steps=args.max_steps,
        log_every=args.log_every, val_every=args.val_every, ckpt_every=args.ckpt_every,
        val_batches=args.val_batches, num_workers=args.num_workers, seed=args.seed,
        device=args.device, max_train_ids=args.max_train_ids, max_val_ids=args.max_val_ids,
    )
    exp = train(cfg)
    print(f"\nDone -> {exp}")


if __name__ == "__main__":
    main()
