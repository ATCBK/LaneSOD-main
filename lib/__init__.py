from lib.InSPyReNet_Res2Net50 import InSPyReNet_Res2Net50

try:
    from lib.InSPyReNet_SwinB import InSPyReNet_SwinB
except ModuleNotFoundError:
    InSPyReNet_SwinB = None
