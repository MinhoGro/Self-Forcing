import torch
import torch.fft as fft
import math


def freq_mix_3d(x, noise, LPF, isCache=False):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    # torch.fft does NOT support bfloat16. Upcast to float32 for FFT, then cast back at the end.
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    noise = noise.to(torch.float32)
    # Ensure LPF is on the same device/dtype for broadcasted multiplies in freq domain
    LPF = LPF.to(device=x.device, dtype=torch.float32)

    # reshape noise& latent
    x = x.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]
    noise = noise.permute(0, 2, 1, 3, 4)

    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    # print(f"LPF shape: {LPF.shape}, x_freq shape: {x_freq.shape}, noise_freq shape: {noise_freq.shape}")
    x_freq_low = x_freq * LPF
    x_freq_high = x_freq * LPF
    noise_freq_low = noise_freq * LPF
    noise_freq_high = noise_freq * HPF
    if isCache:
        x_freq_mixed = x_freq_low # mix in freq domain
    else:
        x_freq_mixed = noise_freq_high + 0.8*x_freq_low + 0.2*noise_freq_low

    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-3, -2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-3, -2, -1)).real

    # Cast back to the original dtype (typically bfloat16) to match the rest of the pipeline
    x_mixed = x_mixed.permute(0, 2, 1, 3, 4) # [B,C,T,H,W] -> [B,T,C,H,W]
    return x_mixed.to(dtype=orig_dtype)


def gaussian_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the gaussian low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = math.exp(-1/(2*d_s**2) * d_square)
    return mask


def butterworth_low_pass_filter(shape, n=4, d_s=0.25, d_t=0.25):
    """
    Compute the butterworth low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        n: order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = 1 / (1 + (d_square / d_s**2)**n)
    return mask


def ideal_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the ideal low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] =  1 if d_square <= d_s*2 else 0
    return mask


def box_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the ideal low pass filter mask (approximated version).

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask

    threshold_s = round(int(H // 2) * d_s)
    threshold_t = round(T // 2 * d_t)

    cframe, crow, ccol = T // 2, H // 2, W //2
    mask[..., cframe - threshold_t:cframe + threshold_t, crow - threshold_s:crow + threshold_s, ccol - threshold_s:ccol + threshold_s] = 1.0

    return mask
