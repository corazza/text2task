from transformers import DataCollatorForLanguageModeling


class ABDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, examples):
        input_ids = []
        labels = []
        for example in examples:
            a, b = example.split(" => ")
            encoded_a = self.tokenizer.encode(
                a.strip(), add_special_tokens=False)
            encoded_b = self.tokenizer.encode(
                b.strip(), add_special_tokens=False)

            # Concatenate the encoded "A" and "B" parts to form the input_ids tensor
            input_ids.append(encoded_a + encoded_b)

            # Encode only the "B" part of the sequence for the labels
            encoded_labels = self.tokenizer.encode(
                b.strip(), add_special_tokens=False)

            # Pad the labels with -100 tokens to match the length of the input_ids tensor
            padding_length = len(input_ids[-1]) - len(encoded_labels)
            labels.append(encoded_labels + [-100] * padding_length)

        input_ids = self._tensorize_batch(input_ids)
        labels = self._tensorize_batch(labels)

        return {"input_ids": input_ids, "attention_mask": input_ids.ne(self.tokenizer.pad_token_id), "labels": labels}
