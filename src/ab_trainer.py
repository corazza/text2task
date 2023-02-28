from torch import nn
from transformers import Trainer
import IPython


class ABTrainer(Trainer):
    def compute_loss_doc(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0]))
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        sep_positions = []
        for input_ids in inputs["input_ids"]:
            sep_pos = (input_ids == self.tokenizer.pad_token_id).nonzero()[0]
            if sep_pos.numel() > 0:
                sep_positions.append(sep_pos[0].item())
            else:
                sep_positions.append(len(input_ids))

        for i, sep_pos in enumerate(sep_positions):
            labels[i, :sep_pos] = -100

        outputs = model(inputs["input_ids"], labels=labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
