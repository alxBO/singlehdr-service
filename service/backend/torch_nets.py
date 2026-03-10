"""PyTorch implementations of SingleHDR networks.
Architectures match the TensorFlow originals exactly for weight compatibility.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Path to the original SingleHDR repo (git submodule)
VENDOR_DIR = os.environ.get('SINGLEHDR_VENDOR_DIR') or os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'vendor', 'SingleHDR')
)


# ============================================================================
# Dequantization-Net (U-Net)
# ============================================================================

class DequantizationNet(nn.Module):
    """Mirrors dequantization_net.py - U-Net encoder-decoder with skip connections."""

    def __init__(self):
        super().__init__()
        # Initial convolutions
        self.conv_in1 = nn.Conv2d(3, 16, 7, padding=3)
        self.conv_in2 = nn.Conv2d(16, 16, 7, padding=3)

        # Down blocks: avg_pool(2) -> conv -> lrelu -> conv -> lrelu
        self.down1_c1 = nn.Conv2d(16, 32, 5, padding=2)
        self.down1_c2 = nn.Conv2d(32, 32, 5, padding=2)
        self.down2_c1 = nn.Conv2d(32, 64, 3, padding=1)
        self.down2_c2 = nn.Conv2d(64, 64, 3, padding=1)
        self.down3_c1 = nn.Conv2d(64, 128, 3, padding=1)
        self.down3_c2 = nn.Conv2d(128, 128, 3, padding=1)
        self.down4_c1 = nn.Conv2d(128, 256, 3, padding=1)
        self.down4_c2 = nn.Conv2d(256, 256, 3, padding=1)

        # Up blocks: upsample(2, bilinear) -> conv -> lrelu -> cat(skip) -> conv -> lrelu
        self.up1_c1 = nn.Conv2d(256, 128, 3, padding=1)
        self.up1_c2 = nn.Conv2d(256, 128, 3, padding=1)   # 128+128 cat
        self.up2_c1 = nn.Conv2d(128, 64, 3, padding=1)
        self.up2_c2 = nn.Conv2d(128, 64, 3, padding=1)    # 64+64 cat
        self.up3_c1 = nn.Conv2d(64, 32, 3, padding=1)
        self.up3_c2 = nn.Conv2d(64, 32, 3, padding=1)     # 32+32 cat
        self.up4_c1 = nn.Conv2d(32, 16, 3, padding=1)
        self.up4_c2 = nn.Conv2d(32, 16, 3, padding=1)     # 16+16 cat

        # Output
        self.conv_out = nn.Conv2d(16, 3, 3, padding=1)

    def _down(self, x, c1, c2):
        x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(c1(x), 0.1)
        x = F.leaky_relu(c2(x), 0.1)
        return x

    def _up(self, x, skip, c1, c2):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.leaky_relu(c1(x), 0.1)
        x = torch.cat([x, skip], dim=1)
        x = F.leaky_relu(c2(x), 0.1)
        return x

    def forward(self, inp):
        x = F.leaky_relu(self.conv_in1(inp), 0.1)
        s1 = F.leaky_relu(self.conv_in2(x), 0.1)
        s2 = self._down(s1, self.down1_c1, self.down1_c2)
        s3 = self._down(s2, self.down2_c1, self.down2_c2)
        s4 = self._down(s3, self.down3_c1, self.down3_c2)
        x = self._down(s4, self.down4_c1, self.down4_c2)
        x = self._up(x, s4, self.up1_c1, self.up1_c2)
        x = self._up(x, s3, self.up2_c1, self.up2_c2)
        x = self._up(x, s2, self.up3_c1, self.up3_c2)
        x = self._up(x, s1, self.up4_c1, self.up4_c2)
        x = torch.tanh(self.conv_out(x))
        return inp + x


# ============================================================================
# Refinement-Net (same U-Net structure, different channels)
# ============================================================================

class RefinementNet(nn.Module):
    """Mirrors refinement_net.py - U-Net, input is 9ch (A_pred + B_pred + C_pred)."""

    def __init__(self):
        super().__init__()
        self.conv_in1 = nn.Conv2d(9, 16, 7, padding=3)
        self.conv_in2 = nn.Conv2d(16, 16, 7, padding=3)

        self.down1_c1 = nn.Conv2d(16, 32, 5, padding=2)
        self.down1_c2 = nn.Conv2d(32, 32, 5, padding=2)
        self.down2_c1 = nn.Conv2d(32, 64, 3, padding=1)
        self.down2_c2 = nn.Conv2d(64, 64, 3, padding=1)
        self.down3_c1 = nn.Conv2d(64, 128, 3, padding=1)
        self.down3_c2 = nn.Conv2d(128, 128, 3, padding=1)
        self.down4_c1 = nn.Conv2d(128, 128, 3, padding=1)
        self.down4_c2 = nn.Conv2d(128, 128, 3, padding=1)

        self.up1_c1 = nn.Conv2d(128, 128, 3, padding=1)
        self.up1_c2 = nn.Conv2d(256, 128, 3, padding=1)   # 128+128
        self.up2_c1 = nn.Conv2d(128, 64, 3, padding=1)
        self.up2_c2 = nn.Conv2d(128, 64, 3, padding=1)    # 64+64
        self.up3_c1 = nn.Conv2d(64, 32, 3, padding=1)
        self.up3_c2 = nn.Conv2d(64, 32, 3, padding=1)     # 32+32
        self.up4_c1 = nn.Conv2d(32, 16, 3, padding=1)
        self.up4_c2 = nn.Conv2d(32, 16, 3, padding=1)     # 16+16

        self.conv_out = nn.Conv2d(16, 3, 3, padding=1)

    def _down(self, x, c1, c2):
        x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(c1(x), 0.1)
        x = F.leaky_relu(c2(x), 0.1)
        return x

    def _up(self, x, skip, c1, c2):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.leaky_relu(c1(x), 0.1)
        x = torch.cat([x, skip], dim=1)
        x = F.leaky_relu(c2(x), 0.1)
        return x

    def forward(self, inp):
        x = F.leaky_relu(self.conv_in1(inp), 0.1)
        s1 = F.leaky_relu(self.conv_in2(x), 0.1)
        s2 = self._down(s1, self.down1_c1, self.down1_c2)
        s3 = self._down(s2, self.down2_c1, self.down2_c2)
        s4 = self._down(s3, self.down3_c1, self.down3_c2)
        x = self._down(s4, self.down4_c1, self.down4_c2)
        x = self._up(x, s4, self.up1_c1, self.up1_c2)
        x = self._up(x, s3, self.up2_c1, self.up2_c2)
        x = self._up(x, s2, self.up3_c1, self.up3_c2)
        x = self._up(x, s1, self.up4_c1, self.up4_c2)
        x = self.conv_out(x)
        return inp[:, :3] + x


# ============================================================================
# Linearization-Net (CrfFeatureNet + AEInvcrfDecodeNet)
# ============================================================================

class _ConvBN(nn.Module):
    """Conv2d + optional BatchNorm + optional ReLU, matching CrfFeatureNet ops."""

    def __init__(self, c_in, c_out, k, s, bias=True, has_bn=False, has_relu=True):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, stride=s, padding=pad, bias=bias)
        self.bn = nn.BatchNorm2d(c_out) if has_bn else None
        self.has_relu = has_relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.has_relu:
            x = F.relu(x)
        return x


class CrfFeatureNet(nn.Module):
    """ResNet50-like feature extractor for camera response function estimation.
    Matches crf_feature_net in linearization_net.py.
    """

    def __init__(self):
        super().__init__()
        # The input has many channels: 3(img) + 6(edges) + 15(hist4) + 27(hist8) + 51(hist16) = 102
        # But CrfFeatureNet receives this pre-concatenated
        # conv1: 7x7, stride 2
        self.conv1 = _ConvBN(102, 64, 7, 2, bias=True, has_bn=True, has_relu=True)

        # pool1
        self.pool1 = nn.MaxPool2d(3, 2, 1)

        # res2a
        self.res2a_b1 = _ConvBN(64, 256, 1, 1, bias=False, has_bn=True, has_relu=False)
        self.res2a_b2a = _ConvBN(64, 64, 1, 1, bias=False, has_bn=True, has_relu=True)
        self.res2a_b2b = _ConvBN(64, 64, 3, 1, bias=False, has_bn=True, has_relu=True)
        self.res2a_b2c = _ConvBN(64, 256, 1, 1, bias=False, has_bn=True, has_relu=False)

        # res2b
        self.res2b_b2a = _ConvBN(256, 64, 1, 1, bias=False, has_bn=True, has_relu=True)
        self.res2b_b2b = _ConvBN(64, 64, 3, 1, bias=False, has_bn=True, has_relu=True)
        self.res2b_b2c = _ConvBN(64, 256, 1, 1, bias=False, has_bn=True, has_relu=False)

        # res2c
        self.res2c_b2a = _ConvBN(256, 64, 1, 1, bias=False, has_bn=True, has_relu=True)
        self.res2c_b2b = _ConvBN(64, 64, 3, 1, bias=False, has_bn=True, has_relu=True)
        self.res2c_b2c = _ConvBN(64, 256, 1, 1, bias=False, has_bn=True, has_relu=False)

        # res3a (stride 2)
        self.res3a_b1 = _ConvBN(256, 512, 1, 2, bias=False, has_bn=True, has_relu=False)
        self.res3a_b2a = _ConvBN(256, 128, 1, 2, bias=False, has_bn=True, has_relu=True)
        self.res3a_b2b = _ConvBN(128, 128, 3, 1, bias=False, has_bn=True, has_relu=True)
        self.res3a_b2c = _ConvBN(128, 512, 1, 1, bias=False, has_bn=True, has_relu=False)

        # res3b
        self.res3b_b2a = _ConvBN(512, 128, 1, 1, bias=False, has_bn=True, has_relu=True)
        self.res3b_b2b = _ConvBN(128, 128, 3, 1, bias=False, has_bn=True, has_relu=True)
        self.res3b_b2c = _ConvBN(128, 512, 1, 1, bias=False, has_bn=True, has_relu=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        # res2a
        b1 = self.res2a_b1(x)
        b2 = self.res2a_b2c(self.res2a_b2b(self.res2a_b2a(x)))
        x = F.relu(b1 + b2)

        # res2b
        b2 = self.res2b_b2c(self.res2b_b2b(self.res2b_b2a(x)))
        x = F.relu(x + b2)

        # res2c
        b2 = self.res2c_b2c(self.res2c_b2b(self.res2c_b2a(x)))
        x = F.relu(x + b2)

        # res3a
        b1 = self.res3a_b1(x)
        b2 = self.res3a_b2c(self.res3a_b2b(self.res3a_b2a(x)))
        x = F.relu(b1 + b2)

        # res3b
        b2 = self.res3b_b2c(self.res3b_b2b(self.res3b_b2a(x)))
        x = F.relu(x + b2)

        # Global average pooling
        x = x.mean(dim=[2, 3])  # [b, 512]
        return x


class AEInvcrfDecodeNet(nn.Module):
    """Decodes a feature vector into a 1024-point inverse CRF using PCA basis."""

    def __init__(self, data_dir=None):
        super().__init__()
        if data_dir is None:
            data_dir = VENDOR_DIR

        self.fc = nn.Linear(512, 11)  # n_p - 1 = 12 - 1 = 11

        # Load PCA basis from invemor.txt
        g0, hinv = self._parse_invemor(os.path.join(data_dir, 'invemor.txt'))
        self.register_buffer('g0', torch.from_numpy(g0).float())       # [1024]
        self.register_buffer('hinv', torch.from_numpy(hinv).float())   # [1024, 11]

    @staticmethod
    def _parse_invemor(path):
        with open(path, 'r') as f:
            lines = [l.strip() for l in f.readlines()]

        def _parse(tag):
            for i, line in enumerate(lines):
                if line == tag:
                    break
            start = i + 1
            vals = []
            for j in range(start, start + 256):  # 1024/4 = 256 lines
                vals += lines[j].split()
            return np.float32(vals)

        g0 = _parse('g0 =')
        hinv = np.stack([_parse('hinv(%d)=' % (i + 1)) for i in range(11)], axis=-1)
        return g0, hinv

    @staticmethod
    def _increase(rf):
        """Ensure monotonically increasing CRF."""
        g = rf[:, 1:] - rf[:, :-1]
        min_g = g.min(dim=-1, keepdim=True).values
        r = F.relu(-min_g)
        new_g = g + r
        new_g = new_g / new_g.sum(dim=-1, keepdim=True)
        new_rf = torch.cumsum(new_g, dim=-1)
        new_rf = F.pad(new_rf, (1, 0), value=0.0)
        return new_rf

    def forward(self, feature):
        # feature: [b, 512]
        w = self.fc(feature)  # [b, 11]

        # PCA reconstruction: invcrf = g0 + hinv @ w
        b = w.shape[0]
        g0 = self.g0.unsqueeze(0).unsqueeze(-1)            # [1, 1024, 1]
        hinv = self.hinv.unsqueeze(0).expand(b, -1, -1)    # [b, 1024, 11]
        w_exp = w.unsqueeze(-1)                              # [b, 11, 1]
        invcrf = g0 + torch.bmm(hinv, w_exp)                # [b, 1024, 1]
        invcrf = invcrf.squeeze(-1)                          # [b, 1024]

        invcrf = self._increase(invcrf)
        return invcrf


class LinearizationNet(nn.Module):
    """Full linearization pipeline: feature extraction + inverse CRF prediction."""

    def __init__(self, data_dir=None):
        super().__init__()
        self.crf_feature_net = CrfFeatureNet()
        self.ae_invcrf_decode_net = AEInvcrfDecodeNet(data_dir)

    @staticmethod
    def _compute_features(img):
        """Compute edge maps and histogram features from input image."""
        # Sobel edges (approximate tf.image.sobel_edges)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=img.dtype, device=img.device).view(1, 1, 3, 3)

        edges = []
        for c in range(3):
            ch = img[:, c:c+1]
            ex = F.conv2d(ch, sobel_x, padding=1)
            ey = F.conv2d(ch, sobel_y, padding=1)
            edges.extend([ex, ey])
        edge_feat = torch.cat(edges, dim=1)  # [b, 6, h, w]

        # Histogram layers
        def histogram_layer(img, max_bin):
            bins = []
            for i in range(max_bin + 1):
                h = F.relu(1.0 - torch.abs(img - i / float(max_bin)) * float(max_bin))
                bins.append(h)
            return torch.cat(bins, dim=1)

        h4 = histogram_layer(img, 4)    # [b, 15, h, w]
        h8 = histogram_layer(img, 8)    # [b, 27, h, w]
        h16 = histogram_layer(img, 16)  # [b, 51, h, w]

        return torch.cat([img, edge_feat, h4, h8, h16], dim=1)  # [b, 102, h, w]

    def forward(self, img):
        features_in = self._compute_features(img)
        feature = self.crf_feature_net(features_in)  # [b, 512]
        invcrf = self.ae_invcrf_decode_net(feature)   # [b, 1024]
        return invcrf


# ============================================================================
# Hallucination-Net (VGG16 encoder-decoder)
# ============================================================================

class HallucinationNet(nn.Module):
    """VGG16-based encoder-decoder for HDR hallucination.
    Matches hallucination_net.py architecture.
    """

    def __init__(self):
        super().__init__()
        # Encoder (VGG16 conv layers)
        self.enc_h1_c1 = nn.Conv2d(3, 64, 3, padding=1)
        self.enc_h1_c2 = nn.Conv2d(64, 64, 3, padding=1)
        self.enc_h2_c1 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc_h2_c2 = nn.Conv2d(128, 128, 3, padding=1)
        self.enc_h3_c1 = nn.Conv2d(128, 256, 3, padding=1)
        self.enc_h3_c2 = nn.Conv2d(256, 256, 3, padding=1)
        self.enc_h3_c3 = nn.Conv2d(256, 256, 3, padding=1)
        self.enc_h4_c1 = nn.Conv2d(256, 512, 3, padding=1)
        self.enc_h4_c2 = nn.Conv2d(512, 512, 3, padding=1)
        self.enc_h4_c3 = nn.Conv2d(512, 512, 3, padding=1)
        self.enc_h5_c1 = nn.Conv2d(512, 512, 3, padding=1)
        self.enc_h5_c2 = nn.Conv2d(512, 512, 3, padding=1)
        self.enc_h5_c3 = nn.Conv2d(512, 512, 3, padding=1)

        # Fully convolutional layer on top of encoder
        self.enc_h6_conv = nn.Conv2d(512, 512, 3, padding=1)
        self.enc_h6_bn = nn.BatchNorm2d(512)

        # Decoder (upsample + conv + BN + skip connections)
        # h1: upsample + conv 512->512 + BN
        self.dec_h1_conv = nn.Conv2d(512, 512, 3, padding=0)  # VALID padding after reflect pad
        self.dec_h1_bn = nn.BatchNorm2d(512)

        # h2: skip(beforepool5, 512) + upsample + conv 512->512 + BN
        self.dec_h2_skip = nn.Conv2d(1024, 512, 1)
        self.dec_h2_conv = nn.Conv2d(512, 512, 3, padding=0)
        self.dec_h2_bn = nn.BatchNorm2d(512)

        # h3: skip(beforepool4, 512) + upsample + conv 512->256 + BN
        self.dec_h3_skip = nn.Conv2d(1024, 512, 1)
        self.dec_h3_conv = nn.Conv2d(512, 256, 3, padding=0)
        self.dec_h3_bn = nn.BatchNorm2d(256)

        # h4: skip(beforepool3, 256) + upsample + conv 256->128 + BN
        self.dec_h4_skip = nn.Conv2d(512, 256, 1)
        self.dec_h4_conv = nn.Conv2d(256, 128, 3, padding=0)
        self.dec_h4_bn = nn.BatchNorm2d(128)

        # h5: skip(beforepool2, 128) + upsample + conv 128->64 + BN
        self.dec_h5_skip = nn.Conv2d(256, 128, 1)
        self.dec_h5_conv = nn.Conv2d(128, 64, 3, padding=0)
        self.dec_h5_bn = nn.BatchNorm2d(64)

        # h6: skip(beforepool1, 64)
        self.dec_h6_skip = nn.Conv2d(128, 64, 1)

        # h7: final conv 64->3 + BN
        self.dec_h7_conv = nn.Conv2d(64, 3, 1)
        self.dec_h7_bn = nn.BatchNorm2d(3)

        # h7 final skip with input (3+3 -> 3)
        self.dec_h7_skip = nn.Conv2d(6, 3, 1)

    def _deconv(self, x, conv, bn):
        """Upsample 2x NN -> reflect pad 1 -> conv (VALID) -> BN."""
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        x = F.relu(conv(x))
        x = bn(x)
        return x

    def _skip_connection(self, decoder_feat, skip_feat, skip_conv):
        """Skip connection: scale skip by 1/255, concat, fuse with 1x1 conv."""
        skip_scaled = skip_feat / 255.0
        x = torch.cat([decoder_feat, skip_scaled], dim=1)
        x = skip_conv(x)
        return x

    def forward(self, x_linear):
        """Input: linearized image [b, 3, h, w] in [0,1] range.
        Output: log-space residual [b, 3, h, w].
        """
        # Scale to [0, 255] as in TF version
        x_in = x_linear * 255.0

        # VGG mean subtraction (BGR order)
        VGG_MEAN = [103.939, 116.779, 123.68]
        r, g, b = x_in[:, 0:1], x_in[:, 1:2], x_in[:, 2:3]
        bgr = torch.cat([b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], dim=1)

        # Encoder
        skip_input = bgr  # will be used for final skip

        h1 = F.relu(self.enc_h1_c1(bgr))
        beforepool1 = F.relu(self.enc_h1_c2(h1))
        p1 = F.max_pool2d(beforepool1, 2, 2)

        h2 = F.relu(self.enc_h2_c1(p1))
        beforepool2 = F.relu(self.enc_h2_c2(h2))
        p2 = F.max_pool2d(beforepool2, 2, 2)

        h3 = F.relu(self.enc_h3_c1(p2))
        h3 = F.relu(self.enc_h3_c2(h3))
        beforepool3 = F.relu(self.enc_h3_c3(h3))
        p3 = F.max_pool2d(beforepool3, 2, 2)

        h4 = F.relu(self.enc_h4_c1(p3))
        h4 = F.relu(self.enc_h4_c2(h4))
        beforepool4 = F.relu(self.enc_h4_c3(h4))
        p4 = F.max_pool2d(beforepool4, 2, 2)

        h5 = F.relu(self.enc_h5_c1(p4))
        h5 = F.relu(self.enc_h5_c2(h5))
        beforepool5 = F.relu(self.enc_h5_c3(h5))
        p5 = F.max_pool2d(beforepool5, 2, 2)

        # Fully-connected conv
        x = self.enc_h6_conv(p5)
        x = F.relu(self.enc_h6_bn(x))

        # Decoder
        x = self._deconv(x, self.dec_h1_conv, self.dec_h1_bn)

        x = self._skip_connection(x, beforepool5, self.dec_h2_skip)
        x = self._deconv(x, self.dec_h2_conv, self.dec_h2_bn)

        x = self._skip_connection(x, beforepool4, self.dec_h3_skip)
        x = self._deconv(x, self.dec_h3_conv, self.dec_h3_bn)

        x = self._skip_connection(x, beforepool3, self.dec_h4_skip)
        x = self._deconv(x, self.dec_h4_conv, self.dec_h4_bn)

        x = self._skip_connection(x, beforepool2, self.dec_h5_skip)
        x = self._deconv(x, self.dec_h5_conv, self.dec_h5_bn)

        x = self._skip_connection(x, beforepool1, self.dec_h6_skip)

        # Final conv
        x = self.dec_h7_conv(x)
        x = self.dec_h7_bn(x)
        # leaky relu with alpha=0 is just identity for positive (no-op since alpha=0)

        x = self._skip_connection(x, skip_input, self.dec_h7_skip)

        return x


# ============================================================================
# Utility: apply inverse CRF
# ============================================================================

def apply_rf_torch(x, rf):
    """Apply response function via interpolation. x: [b,c,h,w], rf: [b,1024]."""
    b, c, h, w = x.shape
    k = rf.shape[1]

    # Flatten spatial dims
    x_flat = x.reshape(b, -1)  # [b, c*h*w]

    # Scale to rf indices
    indices = x_flat.float() * (k - 1)  # [b, c*h*w]

    # Linear interpolation
    idx0 = indices.floor().long().clamp(0, k - 1)
    idx1 = (idx0 + 1).clamp(0, k - 1)
    w1 = indices - idx0.float()
    w0 = 1.0 - w1

    # Gather values from rf
    rf_expanded = rf  # [b, k]
    v0 = torch.gather(rf_expanded, 1, idx0)
    v1 = torch.gather(rf_expanded, 1, idx1)

    result = w0 * v0 + w1 * v1
    return result.reshape(b, c, h, w)
