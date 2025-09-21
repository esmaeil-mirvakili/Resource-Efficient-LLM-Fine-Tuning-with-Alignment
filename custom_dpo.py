from typing import Optional, Dict
from transformers import Trainer
import torch
import torch.nn.functional as F


@torch.no_grad()
def gather_log_probs(logits, labels, ignore_index=-100):
    # shift for next token prediction
    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()

    # mask ignore index
    mask = labels.ne(ignore_index)
    safe_labels = labels.masked_fill(~mask, 0)

    log_probs = F.log_softmax(logits, dim=-1)
    gathered_log_probs = log_probs.gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    gathered_log_probs = gathered_log_probs.masked_fill(~mask, 0.0)

    # sum log probs
    return gathered_log_probs.sum(dim=-1)


def dpo_loss(
    logp_chosen,
    logp_rejected,
    ref_logp_chosen=None,
    ref_logp_rejected=None,
    beta=0.1,
    label_smoothing=0.0,
):
    # if no reference log probs provided, ref free DPO
    if ref_logp_chosen is None:
        ref_logp_chosen = torch.zeros_like(logp_chosen)
    if ref_logp_rejected is None:
        ref_logp_rejected = torch.zeros_like(logp_rejected)
    
    delta_logp = (logp_chosen - logp_rejected) - (ref_logp_chosen - ref_logp_rejected)
    # DPO loss: -log sigmoid(beta * delta_logp)
    # with label smoothing: -(1 - label_smoothing) * log(sigmoid(beta * delta_logp)) - label_smoothing * log(1 - sigmoid(beta * delta_logp))
    pos = F.logsigmoid(beta * delta_logp)
    neg = F.logsigmoid(-beta * delta_logp)
    loss = -(1 - label_smoothing) * pos - label_smoothing * neg
    return loss.mean()


class MyDPOTrainer(Trainer):

    def __init__(
        self,
        ref_model=None,
        chosen_column: str = "chosen",
        rejected_column: str = "rejected",
        precompute_ref_logp: bool = False,
        beta: float=0.1,
        label_smoothing: float=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ref_model = ref_model
        self.beta = beta
        self.chosen_column = chosen_column
        self.rejected_column = rejected_column
        self.precompute_ref_logp = precompute_ref_logp
        self.label_smoothing = label_smoothing
        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        self._ref_cache_train: Optional[Dict[str, torch.Tensor]] = None
        self._ref_cache_eval: Optional[Dict[str, torch.Tensor]] = None
        self._active_cache: Optional[Dict[str, torch.Tensor]] = None

    def _ensure_idx(self, dataset):
        cols = set(getattr(dataset, "column_names", []))
        if "idx" in cols:
            return dataset
        # Map with indices; keep 'idx' persistent through further maps
        return dataset.map(
            lambda ex, i: {"idx": i}, with_indices=True, load_from_cache_file=False
        )

    def _should_precompute(self) -> bool:
        return (
            self.precompute_ref_logp
            and (self.ref_model is not None)
        )

    def _precompute_ref_logprobs(self, dataset):
        assert self.ref_model is not None, "precompute_ref_logprobs requires a reference model"
        dataloader = self.get_eval_dataloader(dataset)
        ref_device = next(self.ref_model.parameters()).device

        all_ref_logp_chosen, all_ref_logp_rejected = {}, {}
        for batch in dataloader:
            batch = {
                k: (v.to(ref_device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            with torch.no_grad():
                ref_out_chosen = self.ref_model(
                    input_ids=batch[f"input_ids_{self.chosen_column}"],
                    attention_mask=batch[f"attention_mask_{self.chosen_column}"],
                )
                ref_logp_chosen = gather_log_probs(
                    ref_out_chosen.logits, batch[f"labels_{self.chosen_column}"]
                ).cpu()

                ref_out_rejected = self.ref_model(
                    input_ids=batch[f"input_ids_{self.rejected_column}"],
                    attention_mask=batch[f"attention_mask_{self.rejected_column}"],
                )
                ref_logp_rejected = gather_log_probs(
                    ref_out_rejected.logits, batch[f"labels_{self.rejected_column}"]
                ).cpu()
            for i, idx in enumerate(batch["idx"].cpu().tolist()):
                all_ref_logp_chosen[idx] = ref_logp_chosen[i : i + 1].item()
                all_ref_logp_rejected[idx] = ref_logp_rejected[i : i + 1].item()
        return {
            "ref_logp_chosen": all_ref_logp_chosen,
            "ref_logp_rejected": all_ref_logp_rejected,
        }

    def train(self, *args, **kwargs):
        if self.train_dataset is None:
            raise ValueError("train_dataset is not set.")
        self.train_dataset = self._ensure_idx(self.train_dataset)
        if self._should_precompute() and self._ref_cache_train is None:
            self._ref_cache_train = self._precompute_ref_logprobs(self.train_dataset)
        self._active_cache = self._ref_cache_train
        try:
            return super().train(*args, **kwargs)
        finally:
            self._active_cache = None

    def evaluate(self, eval_dataset=None, *args, **kwargs):
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            return super().evaluate(eval_dataset, *args, **kwargs)
        dataset = self._ensure_idx(dataset)
        if eval_dataset is not None:
            eval_dataset = dataset
        else:
            self.eval_dataset = dataset
        if self._should_precompute() and self._ref_cache_eval is None:
            self._ref_cache_eval = self._precompute_ref_logprobs(dataset)
        self._active_cache = self._ref_cache_eval
        try:
            return super().evaluate(eval_dataset=eval_dataset, *args, **kwargs)
        finally:
            self._active_cache = None

    def predict(self, test_dataset, **kwargs):
        dataset = self._ensure_idx(test_dataset)
        if self._should_precompute():
            cache = self._precompute_ref_logprobs(dataset)
        else:
            cache = None

        self._active_cache = cache
        try:
            return super().predict(dataset, **kwargs)
        finally:
            self._active_cache = None

    def compute_loss(self, model, inputs, return_outputs=False):
        # policy chosen
        out_chosen = model(
            input_ids=inputs[f"input_ids_{self.chosen_column}"],
            attention_mask=inputs[f"attention_mask_{self.chosen_column}"],
        )
        logp_chosen = gather_log_probs(
            out_chosen.logits, inputs[f"labels_{self.chosen_column}"]
        )
        # policy rejected
        out_rejected = model(
            input_ids=inputs[f"input_ids_{self.rejected_column}"],
            attention_mask=inputs[f"attention_mask_{self.rejected_column}"],
        )
        logp_rejected = gather_log_probs(
            out_rejected.logits, inputs[f"labels_{self.rejected_column}"]
        )
        # reference chosen
        ref_logp_chosen = None
        ref_logp_rejected = None
        if self.ref_model is not None:
            if self.precompute_ref_logp and self._active_cache is not None:
                idx = inputs["idx"].tolist()
                ref_logp_chosen = torch.tensor(
                    [self._active_cache["ref_logp_chosen"][i] for i in idx],
                    device=logp_chosen.device, dtype=logp_chosen.dtype
                )
                ref_logp_rejected = torch.tensor(
                    [self._active_cache["ref_logp_rejected"][i] for i in idx],
                    device=logp_rejected.device, dtype=logp_rejected.dtype
                )
            else:
                with torch.no_grad():
                    ref_out_chosen = self.ref_model(
                        input_ids=inputs[f"input_ids_{self.chosen_column}"],
                        attention_mask=inputs[f"attention_mask_{self.chosen_column}"],
                    )
                    ref_logp_chosen = gather_log_probs(
                        ref_out_chosen.logits, inputs[f"labels_{self.chosen_column}"]
                    )
                    ref_out_rejected = self.ref_model(
                        input_ids=inputs[f"input_ids_{self.rejected_column}"],
                        attention_mask=inputs[f"attention_mask_{self.rejected_column}"],
                    )
                    ref_logp_rejected = gather_log_probs(
                        ref_out_rejected.logits, inputs[f"labels_{self.rejected_column}"]
                    )
        loss = dpo_loss(
            logp_chosen,
            logp_rejected,
            ref_logp_chosen,
            ref_logp_rejected,
            beta=self.beta,
            label_smoothing=self.label_smoothing,
        )
        return (loss, {"loss": loss}) if return_outputs else loss
