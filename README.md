# CoreML3-Conversion-Bug
Contains minimal sample code to reproduce errors that occur when attempting to convert a simple pytorch model to coreml through onnx.

## Steps to reproduce:
1. Run reproduce_issue.py - you'll get a segfault when the mlmodel is used to make predictions
2. Edit reproduce_issue.py by replacing target_ios_verion = '13' with target_ios_verion = '12'.
3. Run reproduce_issue.py and a 100 element vector will be printed out.
