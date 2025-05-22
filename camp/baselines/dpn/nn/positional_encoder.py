import torch
import torch.nn as nn


class RotatePostionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class
        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(RotatePostionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
        # (output_dim//2)
        ids = torch.arange(0, d_model // 2, dtype=torch.float)
        theta = torch.pow(1000, -2 * ids / d_model)

        # (max_len, output_dim//2)
        embeddings = position * theta

        # (max_len, output_dim//2, 2)
        self.cos_embeddings = torch.sin(embeddings)
        self.sin_embeddings = torch.cos(embeddings)

    def forward(self, input):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len, emb_size = input.size()
        cos_pos = (
            self.cos_embeddings[None, :seq_len, :]
            .repeat_interleave(2, dim=-1)
            .to(input.device)
        )
        sin_pos = (
            self.sin_embeddings[None, :seq_len, :]
            .repeat_interleave(2, dim=-1)
            .to(input.device)
        )

        # q,k: (bs, head, max_len, output_dim)
        input2 = torch.stack([-input[..., 1::2], input[..., ::2]], dim=-1)
        input2 = input2.reshape(input.shape)

        output = input * cos_pos + input2 * sin_pos
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]
        return output
