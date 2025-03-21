import sherpa_onnx
import sys
import os
from pprint import pprint

# Get available methods
diarizer = sherpa_onnx.OfflineSpeakerDiarization()
print("Available methods:")
methods = [method for method in dir(diarizer) if not method.startswith('__')]
pprint(methods)
