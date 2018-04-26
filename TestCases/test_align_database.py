import sys
sys.path.append("..")
from UtilsAlign import AlignDatabase, GetAlignFuncByBoundingBox

if __name__ == '__main__':
    
    source = 'imgs\\align_test\\origin'
    target = 'imgs\\align_test\\aligned'
    
    F = GetAlignFuncByBoundingBox(output_size=224, margin=24)
    AlignDatabase(source, target, align_func=F)
    
