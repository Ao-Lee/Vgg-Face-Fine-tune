from align import AlignDatabase, GetAlignFuncByBoundingBox

if __name__ == '__main__':
    
    source = 'E:\\DM\\ARFace'
    target = 'E:\\DM\\ARFace_aligned'

    F = GetAlignFuncByBoundingBox(output_size=224, margin=24)
    AlignDatabase(source, target, align_func=F)
    
