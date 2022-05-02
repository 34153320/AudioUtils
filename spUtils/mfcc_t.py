# mfcc in torch version 

import torch
import numpy as np
from .torch_fbank import fBank
import torch.nn.functional as tf

class MFCCT(torch.nn.Module):
     def __init__(self, win_len, hop_size, fft_len, sr,
                  win_type='hann', power=2.0, n_mel=128, n_mfcc=13,
                  dict_type=1, norm='ortho', ref=1.0,
                  top_db=80.0, center=True, enframe_mode='continue'):
         """
          Torch version MFCC using torch filterbank
          Implement of MFCC using 1D convolution and 1D transpose convolution
          Code was modified from 
          https://github.com/echocatzh/torch-mfcc/blob/master/torch_mfcc/torch_mfcc.py
          
          Params:
          
         """
         super().__init__()
         self.fbank = fBank()
         self.dict_type = dict_type

     def _init_kernel_(self):
         """
          Return the Discrete Cosine Transform of arbitrary type sequence x.
          Four types of DCT can be found here:
          https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html

          Returns:
              tensors: The DCT transform matrix
         """
         numMels = torch.arrange(self.n_mel)[:, None].float()
         numMFCC = torch.arrange(self.n_mfcc)[:, None].float()
         dict_kernel = None
         if self.dict_type == 2:
             dict_kernel = 2.0 * \
                   torch.cos(np.pi*torch.matmul(2.*numMels+1.0, numMFCC.T)/2./self.n_mel)
             if self.norm == 'ortho':
                dict_kernel[:, 0] *= np.sqrt(1/self.n_mel/4.)
                dict_kernel[:, 1:] *= np.sqrt(1/self.n_mel/2.)
         elif self.dict_type == 3:
             dict_kernel = 2.0*torch.cos(np.pi*torch.matmul(numMels, 2*numMFCC.T+1)/(2*self.n_mel))
             dict_kernel[0, :] = 0.
             if self.norm == 'ortho':
                dict_kernel *= np.sqrt(1/(2*self.n_mel))
         elif self.dict_type == 4:


         else:
             raise RuntimeError("Type {} is not supported".format(self.dict_type))