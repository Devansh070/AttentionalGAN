import torch
import torch.nn as nn


def conv1x1(in_planes, out_planes):
    """1x1 convolution with no padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)


def func_attention(query, context, gamma1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = context.transpose(1, 2).contiguous()

    # Get attention
    attn = torch.bmm(contextT, query)  # batch x sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = torch.nn.functional.softmax(attn, dim=-1)  # Apply softmax

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL).transpose(1, 2).contiguous()

    # Eq. (9)
    attn = attn.view(batch_size * queryL, sourceL) * gamma1
    attn = torch.nn.functional.softmax(attn, dim=-1)
    attn = attn.view(batch_size, queryL, sourceL)

    # --> batch x ndf x queryL
    attnT = attn.transpose(1, 2).contiguous()
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)


class GlobalAttentionGeneral(nn.Module):
    def __init__(self, idf, cdf):
        super(GlobalAttentionGeneral, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.sm = nn.Softmax(dim=-1)  # Use a fixed dimension
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context):
        """
        input: batch x idf x ih x iw (queryL=ihxiw)
        context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)
        targetT = target.transpose(1, 2).contiguous()

        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        sourceT = context.unsqueeze(3)
        sourceT = self.conv_context(sourceT).squeeze(3)

        # Get attention
        attn = torch.bmm(targetT, sourceT)  # batch x queryL x sourceL
        attn = attn.view(batch_size * queryL, sourceL)
        if self.mask is not None:
            mask = self.mask.repeat(queryL, 1)
            attn = attn.masked_fill(mask, -float('inf'))
        attn = self.sm(attn)
        attn = attn.view(batch_size, queryL, sourceL)

        # --> batch x sourceL x queryL
        attn = attn.transpose(1, 2).contiguous()

        # (batch x idf x sourceL)(batch x sourceL x queryL) --> batch x idf x queryL
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn
