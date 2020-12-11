import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceAttention(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.
    """

    def __init__(self, dim1, dim2):
        super(FaceAttention, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.linear1 = nn.Linear(dim1 + dim2, dim2)
        self.linear2 = nn.Linear(dim2, 1, bias=False)
        self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, fc_feat, img_feat, label_feat, face, face_mask):
        """
        Arguments:
            hidden_state {Variable} -- batch_size x dim
            encoder_outputs {Variable} -- batch_size x seq_len x dim
        Returns:
            Variable -- context vector of size batch_size x dim
        """
        eps = 1e-5
        batch_size, seq_len, _ = face.size()
        label_feat = label_feat.unsqueeze(1).repeat(1, seq_len, 1)

        inputs = torch.cat((face, fc_feat, img_feat, label_feat),
                           2).view(-1, self.dim1 + self.dim2)
        o = self.linear2(torch.tanh(self.linear1(inputs)))
        e = o.view(batch_size, seq_len)
        alpha = F.softmax(e, dim=1)

        alpha = alpha * face_mask.view(-1, seq_len).float()
        alpha = alpha / (alpha.sum(1, keepdim=True) + eps)  # normalize to 1

        context = torch.bmm(alpha.unsqueeze(1), face).squeeze(1)
        return alpha, context

class FaceAttention2(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.
    """

    def __init__(self, dim1, dim2):
        super(FaceAttention2, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.linear1 = nn.Linear(dim1 + dim2, dim2)
        self.linear2 = nn.Linear(dim2, 1, bias=False)
        self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, fc_feat, img_feat, label_feat, memory_feat, face, face_mask):
        """
        Arguments:
            hidden_state {Variable} -- batch_size x dim
            encoder_outputs {Variable} -- batch_size x seq_len x dim
        Returns:
            Variable -- context vector of size batch_size x dim
        """
        eps = 1e-5

        batch_size, seq_len, _ = face.size()
        label_feat = label_feat.unsqueeze(1).repeat(1, seq_len, 1)
        memory_feat = memory_feat.unsqueeze(1).repeat(1, seq_len, 1)

        inputs = torch.cat((face, fc_feat, img_feat, label_feat, memory_feat),
                           2).view(-1, self.dim1 + self.dim2)
        o = self.linear2(torch.tanh(self.linear1(inputs)))
        e = o.view(batch_size, seq_len)
        alpha = F.softmax(e, dim=1)

        alpha = alpha * face_mask.view(-1, seq_len).float()
        alpha = alpha / (alpha.sum(1, keepdim=True) + eps)  # normalize to 1

        context = torch.bmm(alpha.unsqueeze(1), face).squeeze(1)
        return alpha, context

class FaceAttention3(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.
    """

    def __init__(self, dim1, dim2):
        super(FaceAttention3, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.linear1 = nn.Linear(dim1 + dim2, dim2)
        self.linear2 = nn.Linear(dim2, 1, bias=False)
        self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, fc_feat, img_feat, label_feat, memory, face, face_mask):
        """
        Arguments:
            hidden_state {Variable} -- batch_size x dim
            encoder_outputs {Variable} -- batch_size x seq_len x dim
        Returns:
            Variable -- context vector of size batch_size x dim
        """
        eps = 1e-5

        batch_size, seq_len, _ = face.size()
        final_seq_len = seq_len * memory.size(1)
        label_feat = label_feat.unsqueeze(1).repeat(1, seq_len, 1)
        inputs = None
        for i in range(memory.size(1)):
            memory_feat = memory[:,i].unsqueeze(1).repeat(1, seq_len, 1)
            dot_feat = memory_feat
            att_feat = torch.cat((face, fc_feat, img_feat, label_feat, dot_feat),2).unsqueeze(1)
            if inputs is None:
                inputs = att_feat
            else:
                inputs = torch.cat((inputs, att_feat), dim=1)
        inputs = inputs.view(batch_size, final_seq_len, -1)
        inputs = inputs.view(-1, self.dim1 + self.dim2).unsqueeze(1)
        o = self.linear2(torch.tanh(self.linear1(inputs)))
        e = o.view(batch_size, final_seq_len)
        alpha = F.softmax(e, dim=1)

        final_face_mask = face_mask.repeat(1, memory.size(1))

        alpha = alpha * final_face_mask.float()
        alpha = alpha / (alpha.sum(1, keepdim=True) + eps)  # normalize to 1

        context = torch.bmm(alpha.unsqueeze(1), face.repeat(1,memory.size(1),1)).squeeze(1)
        return alpha, context
