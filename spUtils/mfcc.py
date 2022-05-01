# this tool is used for multiresolution MFCC calculation 
import numpy as np

PI = 3.14159265358979323846264338327

class MFCC():
      def __init__(self, spectrogram, 
                         samplingRate,
                         numFilters, 
                         binSize,
                         mThCoeff,
                         frequencyBand,
                         filterBand) -> float:
          """
            params: 
            spectrogram: array of floats containing the results of FFT computation. The data is assumed to be 
                         real maganitude. 
            samplingRate: the rate of the original time-series sampled at (i.e., 16000 Hz)
            numFilters:  the number of filters to use in the computation. Recommended values include 20, 32, 48
            binSize: the size of the spectrogram array, usually a power of 2
            mThCoeff: the mth MFCC coefficient to compute.
          """
          self.spectogram = spectrogram
          self.samplingRate = samplingRate
          self.numFilters = numFilters
          self.binSize = binSize
          self.mThCoeff = mThCoeff
          self.frequencyBand = frequencyBand
          self.filterBand = filterBand

      def get_mfcc_coefficient(self):
          # compute the mth MFCC coefficients

          if self.mThCoeff >= self.numFilters:
             return 0
          
          outerSum = 0
          result = self.normalization_factor(self.numFilters, self.mThCoeff)
         
          for idxFilter in range(1, self.numFilters+1):
              innerSum = 0
              for idxBin in range(self.binSize-1):
                  innerSum += np.abs(self.spectogram[idxBin] * self._get_filter_parameter(self.samplingRate, \
                                                                                   self.binSize, idxBin, idxFilter))
              if innerSum > 0:
                  innerSum =  np.log(innerSum)
              innerSum = innerSum * np.cos(((self.mThCoeff * PI) / self.numFilters) * (idxFilter-0.5))   
              outerSum += innerSum

          return result * outerSum   
    
      def normalization_factor(self):
          normalizationFactor = 0
          if self.mThCoeff == 0: # first component of the MFCC 
              normalizationFactor = np.sqrt(1.0/self.numFilters)
          else:
              normalizationFactor = np.sqrt(2.0/self.numFilters)
          return normalizationFactor

      def _get_filter_parameter(self):
          # compute the filter parameter for the specified frequency and filter bands
          # Used as internal computation only 
          filterParameter = 0.0
          boundary = (self.frequencyBand * self.samplingRate) / self.binSize
          prevCenterFrequency = self._get_center_frequency(self.filterBand - 1)
          thisCenterFrequency = self._get_center_frequency(self.filterBand)
          nextCenterFrequency = self._get_center_frequency(self.filterBand + 1)

          if boundary >=0 & boundary < prevCenterFrequency:
              filterParameter = 0.0
          elif boundary >= prevCenterFrequency & boundary < thisCenterFrequency:
              filterParameter = (boundary - prevCenterFrequency) / (thisCenterFrequency - prevCenterFrequency)
              filterParameter *= self._get_magnitude_factor(self.filterBand)
          elif boundary >= thisCenterFrequency & boundary < nextCenterFrequency:
              filterParameter = (boundary -nextCenterFrequency) / (thisCenterFrequency - nextCenterFrequency)
              filterParameter *= self._get_magnitude_factor(self.filterBand)
          elif boundary >= nextCenterFrequency & boundary < self.samplingRate:
              filterParameter = 0.0
          return filterParameter 

      def _get_magnitude_factor(self):
          # compute the band-dependent magnitude factor for the given filter band
          # 
          magnitudeFactor = 0.0
          if self.filterBand >= 1 & self.filterBand <= 14:
              magnitudeFactor = 0.015
          elif self.filterBand >= 15 & self.filterBand <=48:
              magnitudeFactor = 2.0 / (self._get_center_frequency(self.filterBand + 1) - self._get_center_frequency(self.filterBand - 1))
          return magnitudeFactor
      
      def _get_center_frequency(self):
          # compute the center frequency (fc) of the specified filter band l
          # filters are specified so the center frequencies are equally spaced on the mel
          # scale. 
          centerFrequency  = 0.0
          if self.filterBand == 0:
              centerFrequency = 0
          elif self.filterBand >= 1 & self.filterBand <=14:
              centerFrequency = (200.0 * self.filterBand) / 3.0
          else:
              exponent = self.filterBand - 14.0
              centerFrequency = np.power(1.0711703, exponent)
              centerFrequency *= 1073.4

          return centerFrequency



